"""Local Whisper transcription and the voice HTTP endpoint."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
import os
import time
from urllib.parse import urlsplit

from aiohttp import web

# from .voice_commands import interpret_command
from .voice_commands import interpret_command, warmup_command_matcher

LOG = logging.getLogger("yolo_live.voice")
VOICE = web.AppKey("voice_service", object)

RATE = 16000
MAX_SECONDS = 12
MAX_BYTES = 2_000_000

AUDIO_FORMATS = {
    "audio/webm": "matroska",
    "audio/ogg": "ogg",
    "audio/mp4": "mov",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


class VoiceBusy(Exception):
    pass


def decode_audio(payload, container_format):
    import av
    import numpy as np

    pieces = []
    sample_count = 0
    duration = 0.0

    def append(frame):
        nonlocal sample_count
        values = frame.to_ndarray().reshape(-1)
        sample_count += values.size
        if sample_count > MAX_SECONDS * RATE:
            raise ValueError("Keep recordings shorter than 12 seconds.")
        pieces.append(values.copy())

    try:
        with av.open(BytesIO(payload), format=container_format) as source:
            if not source.streams.audio:
                raise ValueError("The recording contains no audio.")

            resampler = av.AudioResampler(
                format="fltp", layout="mono", rate=RATE,
            )

            for frame in source.decode(audio=0):
                if not frame.sample_rate or frame.sample_rate > 96000:
                    raise ValueError("Unsupported audio sample rate.")
                if len(frame.layout.channels) > 2:
                    raise ValueError("Send mono or stereo microphone audio.")

                duration += frame.samples / frame.sample_rate
                if duration > MAX_SECONDS:
                    raise ValueError("Keep recordings shorter than 12 seconds.")

                for converted in resampler.resample(frame):
                    append(converted)

            for converted in resampler.resample(None):
                append(converted)

    except av.error.FFmpegError as error:
        raise ValueError("The microphone recording could not be decoded.") from error

    if not pieces:
        raise ValueError("The recording is empty.")

    audio = np.concatenate(pieces).astype(np.float32, copy=False)
    if not np.isfinite(audio).all():
        raise ValueError("Invalid audio samples.")

    return audio


class SpeechBackend:
    def __init__(self, device):
        self.device = device
        self.model_id = os.environ.get(
            "VOICE_MODEL", "openai/whisper-large-v3-turbo",
        )

    def load(self):
        import torch
        from silero_vad import load_silero_vad
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the speech model.")

        self.dtype = (
            torch.float16 if self.device.startswith("cuda") else torch.float32
        )

        LOG.info("Loading speech models: %s on %s", self.model_id, self.device)
        self.vad = load_silero_vad().eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            use_safetensors=True,
            attn_implementation="sdpa",
        ).to(self.device).eval()
        LOG.info("Speech models ready.")
        warmup_command_matcher()


    def interpret(self, payload, container_format, language):
        import torch
        from silero_vad import get_speech_timestamps

        started = time.perf_counter()
        audio = decode_audio(payload, container_format)

        with torch.inference_mode():
            spans = get_speech_timestamps(
                torch.from_numpy(audio),
                self.vad,
                sampling_rate=RATE,
                min_speech_duration_ms=100,
                min_silence_duration_ms=300,
                speech_pad_ms=150,
            )

            if not spans:
                result = interpret_command("")
            else:
                # Preserve internal pauses, corrections, and negations.
                audio = audio[spans[0]["start"]:spans[-1]["end"]]

                inputs = self.processor(
                    audio,
                    sampling_rate=RATE,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                inputs = {
                    name: tensor.to(
                        device=self.device,
                        dtype=(
                            self.dtype
                            if tensor.is_floating_point() else tensor.dtype
                        ),
                    )
                    for name, tensor in inputs.items()
                }

                tokens = self.model.generate(
                    **inputs,
                    task="transcribe",
                    language=None if language == "auto" else "en",
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=128,
                    condition_on_prev_tokens=False,
                )
                transcript = self.processor.batch_decode(
                    tokens, skip_special_tokens=True,
                )[0].strip()
                result = interpret_command(transcript)

        result["processing_ms"] = round(
            (time.perf_counter() - started) * 1000, 1,
        )
        return result


class VoiceService:
    def __init__(self, device):
        self.backend = SpeechBackend(device)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="voice",
        )
        self.pending = None
        self.closing = False

    async def load(self):
        await asyncio.get_running_loop().run_in_executor(
            self.executor, self.backend.load,
        )

    async def interpret(self, payload, container_format, language):
        if self.closing or (
            self.pending is not None and not self.pending.done()
        ):
            raise VoiceBusy()

        # No await between admission and submission: requests cannot queue
        # behind an already-running speech inference.
        job = asyncio.get_running_loop().run_in_executor(
            self.executor,
            self.backend.interpret,
            payload,
            container_format,
            language,
        )
        self.pending = job

        # Retrieve errors even if the HTTP caller disconnects.
        job.add_done_callback(
            lambda completed: (
                completed.exception() if not completed.cancelled() else None
            )
        )
        # Disconnecting cannot cancel the underlying GPU operation.
        return await asyncio.shield(job)

    async def close(self):
        self.closing = True
        await asyncio.to_thread(
            self.executor.shutdown, wait=True, cancel_futures=True,
        )


def reply(request_id, status, message="", http_status=200, **fields):
    return web.json_response(
        {
            "request_id": request_id,
            "status": status,
            "message": message,
            **fields,
        },
        status=http_status,
        headers={"Cache-Control": "no-store"},
    )


async def voice_request(request):
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc.lower() != request.host.lower():
        raise web.HTTPForbidden(text="Use the microphone on this app's page.")

    if request.headers.get("X-Voice-Request") != "1":
        raise web.HTTPForbidden(text="Missing voice request header.")

    request_id = request.headers.get("X-Request-ID", "")
    if not request_id or len(request_id) > 80:
        raise web.HTTPBadRequest(text="Invalid request ID.")

    language = request.query.get("language", "en")
    if language not in ("en", "auto"):
        raise web.HTTPBadRequest(text="Unsupported language mode.")

    container_format = AUDIO_FORMATS.get(request.content_type)
    if container_format is None:
        return reply(
            request_id, "invalid_audio",
            "Use WebM, Ogg, MP4, or WAV audio.", 415,
        )

    try:
        payload = bytearray()
        async for chunk in request.content.iter_chunked(65536):
            payload.extend(chunk)
            if len(payload) > MAX_BYTES:
                return reply(
                    request_id, "invalid_audio",
                    "The recording is too large.", 413,
                )

        if not payload:
            raise ValueError("The recording is empty.")

        result = await request.app[VOICE].interpret(
            bytes(payload), container_format, language,
        )
        return web.json_response(
            {"request_id": request_id, **result},
            headers={"Cache-Control": "no-store"},
        )

    except VoiceBusy:
        return reply(
            request_id, "busy",
            "Speech processing is busy. Please try again shortly.", 429,
        )
    except ValueError as error:
        return reply(request_id, "invalid_audio", str(error), 400)
    except Exception:
        LOG.exception("Speech processing failed")
        return reply(
            request_id, "error",
            "Speech processing failed. Check the server terminal.", 500,
        )


def install_voice(app, device):
    app[VOICE] = VoiceService(device)
    app.router.add_post("/api/voice/interpret", voice_request)

    async def lifecycle(app):
        try:
            await app[VOICE].load()
            yield
        finally:
            await app[VOICE].close()

    app.cleanup_ctx.append(lifecycle)