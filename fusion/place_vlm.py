"""Where is the user? room / restroom / hallway / lobby / stairwell, from the camera, by the VLM.

This replaces the old rule scoring (scene_memory.classify_place) and the text-LLM guesses
(intent_llm.plan_place, search_llm.plan_target). There are no rule-based guesses any more:
if the VLM is off or fails, the place is "unknown" and each feature takes its unknown path.

Model: Qwen3-VL-8B-Instruct-FP8, served by HP Z Runtime (vLLM, OpenAI-compatible API):
    zrt serve hf:Qwen/Qwen3-VL-8B-Instruct-FP8 --label vlm --gpu-memory-fraction 0.25 \
        -- --max-model-len 8192 --limit-mm-per-prompt '{"image":4}'
    -> http://127.0.0.1:8080/v1, model "vlm"   (fusion_server.py --vlm-url / --vlm-model / --no-vlm)

One PlaceContext per phone (fusion_server.py):
    add_frame(jpeg, heading, now)   every frame: keeps the newest frame per 45° direction (last 12 s)
    tick(now, hints)                every frame: every 3 s asks the VLM in the background (never blocks)
    current(now)                    the last answer (held 20 s); for code that must not wait
    get(now, hints, wait)           same, but asks now and waits up to `wait` s if the answer is old
The VLM answers ONE word (room / restroom / hallway / lobby / stairwell / unknown): up to 4 small
frames (384 px), a short prompt, max 3 output tokens, no JSON, no score, no reason - for latency.
current()/get() return {"place", "planner_place"}; planner_place maps lobby / stairwell -> hallway
(open public areas) for the exit and search planners.
"""
import base64
import json
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from scene_memory import SceneMemory

PLACES = ("room", "restroom", "hallway", "lobby", "stairwell", "unknown")
PLANNER_PLACE = {"room": "room", "restroom": "restroom", "hallway": "hallway", "lobby": "hallway",
                 "stairwell": "hallway", "unknown": "unknown"}
REFRESH_S = 3.0      # ask the VLM this often (in the background)
FRESH_S = 5.0        # get(): an answer newer than this is used without asking again
HOLD_S = 20.0        # an answer counts for this long
KEEP_S = 12.0        # frames older than this are not sent
CONFIRM = 2          # a new place replaces a recent one after this many answers in a row (1 = at once)
MAX_IMAGES = 4
BIN_DEG = 45         # one frame per direction slice, so the VLM sees around the user, not 4 copies
IMAGE_MAX_SIDE = 384 # frames are shrunk to this before sending: ~110 image tokens each instead of ~300
MAX_TOKENS = 3       # the answer is one word

# Short on purpose (latency): the answer is ONE word, parsed below; no JSON, no score, no reason.
SYSTEM = """Camera frames from a phone carried by a blind person indoors, taken seconds apart from one spot,
facing different directions. What kind of place is it?
room = classroom, office, lab, meeting room, kitchen (enclosed, with furniture)
restroom = toilets, urinals, stalls
hallway = corridor, long and narrow, doors along the walls
lobby = large open entrance, waiting area, atrium
stairwell = stairs or a landing
unknown = too dark, blurry or close to a wall to tell
Answer with exactly one word: room, restroom, hallway, lobby, stairwell or unknown."""


def scene_hints(scene, now):
    """Text hints for the VLM from the scene memory (objects, sign text, LiDAR free-space size)."""
    if scene is None:
        return ""
    try:
        return SceneMemory.describe(scene.summary(now))
    except Exception:  # noqa: BLE001
        return ""


def parse_place(text):
    """The first allowed place word in the reply ("Hallway." -> "hallway"), or None."""
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if w in PLACES:
            return w
    return None


def shrink(jpeg):
    """Smaller image = fewer image tokens = faster. Falls back to the original on any problem."""
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        scale = IMAGE_MAX_SIDE / max(h, w)
        if scale >= 1:
            return jpeg
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else jpeg
    except Exception:  # noqa: BLE001
        return jpeg


class VLMClient:
    """OpenAI-style /chat/completions with images (vLLM). ask() returns one place word or None."""

    def __init__(self, url="http://127.0.0.1:8080/v1", model="vlm", timeout=6.0, api_key=None):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def _request(self, path, body=None, timeout=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read())

    def check(self):
        try:
            ids = [m.get("id") for m in self._request("/models", timeout=3).get("data", [])]
            ok = self.model in ids
            print(f"VLM: {self.url} models={ids} -> {'ready' if ok else 'model ' + repr(self.model) + ' not served'}")
            return ok
        except Exception as e:  # noqa: BLE001
            print(f"VLM: {self.url} unreachable ({e}); the place stays 'unknown'")
            return False

    def ask(self, jpegs, hints=""):
        content = [{"type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
                   for j in jpegs]
        text = (f"Hints: {hints}\n" if hints else "") + "One word:"
        body = {"model": self.model, "temperature": 0, "max_tokens": MAX_TOKENS,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": content + [{"type": "text", "text": text}]}]}
        r = self._request("/chat/completions", body)
        return parse_place(r["choices"][0]["message"]["content"])


class PlaceContext:
    """The one shared answer to "where is the user?" for one phone connection."""

    _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")

    def __init__(self, client=None):
        self.client = client
        self.frames = {}          # direction slice -> (time, jpeg)
        self.result = None        # validated answer + "at"
        self.pending = None       # Future of the running VLM call
        self.last_ask = -1e9
        self.candidate = None     # a new place seen, waiting to be confirmed
        self.seen_count = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ frames
    def add_frame(self, jpeg, heading, now):
        if self.client is None or not jpeg:
            return
        b = int((heading % 360) // BIN_DEG)
        with self._lock:
            self.frames[b] = (now, jpeg)
            self.frames = {k: v for k, v in self.frames.items() if now - v[0] <= KEEP_S}

    def _pick(self):
        with self._lock:
            newest = sorted(self.frames.values(), key=lambda v: -v[0])[:MAX_IMAGES]
        return [j for _, j in reversed(newest)]

    # ------------------------------------------------------------------ asking
    def tick(self, now, hints=None):
        """Every frame. Asks the VLM in the background every REFRESH_S. `hints` may be a callable."""
        if self.client is None or now - self.last_ask < REFRESH_S:
            return
        if self.pending is not None and not self.pending.done():
            return
        self._submit(now, hints)

    def _submit(self, now, hints):
        jpegs = self._pick()
        if not jpegs:
            return None
        self.last_ask = now
        text = hints() if callable(hints) else (hints or "")
        self.pending = self._pool.submit(self._run, jpegs, text, now)
        return self.pending

    def _run(self, jpegs, hints, asked_at):
        t0 = time.perf_counter()
        try:
            place = self.client.ask([shrink(j) for j in jpegs], hints)
        except Exception as e:  # noqa: BLE001  (timeout, server down)
            print(f"VLM place failed: {type(e).__name__}: {e}")
            return None
        ms = round((time.perf_counter() - t0) * 1000)
        if place is None:
            print(f"VLM place: reply was not a place word ({ms} ms)")
            return None
        self._accept(place, asked_at)
        cur = self.result["place"] if self.result else "unknown"
        print(f"VLM place: {place} ({len(jpegs)} images, {ms} ms) -> {cur}")
        return place

    def _accept(self, place, at):
        """Steady without scores: a recent place changes only after CONFIRM answers in a row
        for the new place; "unknown" never replaces a recent place."""
        with self._lock:
            old = self.result
            recent = old is not None and at - old["at"] <= HOLD_S and old["place"] != "unknown"
            if recent and place == "unknown":
                return
            if recent and place != old["place"]:
                self.seen_count = self.seen_count + 1 if self.candidate == place else 1
                self.candidate = place
                if self.seen_count < CONFIRM:
                    return                        # wait for the next answer
            self.candidate, self.seen_count = None, 0
            self.result = {"place": place, "at": at}

    # ------------------------------------------------------------------ reading
    def current(self, now):
        r = self.result
        if r is None or now - r["at"] > HOLD_S:
            return {"place": "unknown", "planner_place": "unknown"}
        return {"place": r["place"], "planner_place": PLANNER_PLACE[r["place"]]}

    def get(self, now, hints=None, wait=4.0):
        """The current answer; if it is older than FRESH_S, asks now and waits up to `wait` s."""
        r = self.result
        if self.client is not None and (r is None or now - r["at"] > FRESH_S):
            fut = self.pending if (self.pending is not None and not self.pending.done()) else self._submit(now, hints)
            if fut is not None:
                try:
                    fut.result(timeout=wait)
                except Exception:  # noqa: BLE001  (timeout: use whatever is there)
                    pass
        return self.current(time.monotonic())
