# Indoor voice update

Prepared against the supplied live-yolo-webui-v2 source on 2026-09-23.
This ZIP contains complete replacement files, not snippets or a standalone app.
The original project folders were not modified. No app, model, compiler, or tests
were executed during preparation. Review and validate this build before use.

## Install on your GB10

1. Stop your running YOLO server and back up your current project.
2. Copy this ZIP to the project directory containing run_yolo.py.
3. Extract its contents INTO that directory, merging yolo_live/ and replacing the
   six existing files listed below. Do not replace the entire yolo_live directory;
   it still needs your existing appearance.py, voice_commands.py, direction.py,
   voice_direction.py, ocr.py, and __init__.py.
4. Start with your existing arguments and add --voice, for example:

   ```bash
   conda activate indoor-nav
   python run_yolo.py --port 8091 --voice
   ```

   Preserve any --http, --weights, --device, certificate, or OCR arguments you
   already use. This update keeps your YOLOE, DINOv2, Whisper, Silero, MiniLM and
   PP-OCRv5 model choices. No additional Python packages or model downloads are
   introduced by this update beyond your working v2 requirements.
5. Reload the browser page completely. The three scripts use a new cache version.
6. Activate Enable voice with a screen reader, keyboard, or touch. Grant the
   browser microphone permission. Speak after the short tone: "Find a chair".
   Camera permission is requested when a camera command first needs it.

Optional extraction command, run yourself from the project root:

```bash
python -m zipfile -e indoor-voice-update.zip .
```

The archive root contains yolo_live/, tests_voice_update/, this guide, and a
baseline hash manifest. It does not contain models, environments, or recordings.
SOURCE_BASELINE_SHA256.json records the supplied source version this update was
built against; it is not a dependency lockfile or an installer.

## Files

| Action | File | Purpose |
| --- | --- | --- |
| Replace | yolo_live/server.py | Expose voice capability/classes in health; support releasing a target |
| Replace | yolo_live/engine.py | Skip appearance extraction when no target is selected |
| Replace | yolo_live/sp2text.py | Dispatch recognized text through the new intent router |
| Replace | yolo_live/static/app.js | Camera/control bridge, fresh results, OCR snapshots, target cancellation |
| Replace | yolo_live/static/voice.js | Recording, dialogue, voice controls and spoken state changes |
| Replace | yolo_live/static/index.html | Load the updated browser modules |
| Add | yolo_live/voice_intents.py | Explicit control intents before the existing semantic target matcher |
| Add | yolo_live/static/speech.js | Speech priority, expiry, interruption and repetition |
| Add | yolo_live/static/voice.css | Accessible touch targets and visible focus |
| Restore if missing | yolo_live/static/styles.css | Original v2 theme, layout and OCR styling; included unchanged |
| Add | tests_voice_update/ | Model-free checks supplied for you to run |

## Commands and their behavior

| Say | Result |
| --- | --- |
| "Find a chair"; "Hi, can you find a chair for me please?" | Start camera if needed, select chair, begin a new target session |
| "Find the chair on my left" | Apply the side constraint to initial target selection |
| "The left one" | Answer a recent multiple-object question; context expires after 20 seconds |
| "The second one" | Choose from a recent spoken list of up to three object alternatives |
| "Start camera" | Connect camera without selecting an arbitrary initial target |
| "Stop camera" | Close camera and release its target; voice remains available |
| "Cancel target" | Clear the backend appearance reference; keep camera and OCR available |
| "Pause" / "Resume" | Pause/resume automatic spoken directions; vision continues remembering the target |
| "Repeat" | Repeat the last still-valid message; expired guidance falls back to current status |
| "Status" | Describe the current session and current target state |
| "Read text" / "Read the sign" | Read a fresh OCR snapshot, enabling OCR if available |
| "Enable text reading" / "Disable text reading" | Toggle OCR without clearing the target reference |
| "Help" / "What can I say?" | Explain the commands |
| "List targets" / "What can you find?" | Read the configured detector vocabulary |
| "Keep listening" / "Hands free on" | Enable foreground listening between announcements |
| "Stop listening" / "Hands free off" | Return to the Speak command button |
| "Stop" / "Stop everything" | Turn off camera and microphone, release target and disable hands-free mode |

Control commands use exact normalized phrases, including optional greetings and
politeness. Negated or combined requests such as "do not stop" or "stop and find
a chair" do not execute a control. Other target requests still go through your
existing rules and semantic model, with its existing ambiguity thresholds.
A side-only answer needs a current clarification; it never silently changes an
already locked object's identity. New explicit target requests can replace any
pending clarification. Status/help/read commands do not create new target locks.

## Using it without looking at the UI

- Enable voice once with an accessible button; permissions remain browser/OS UI.
- Say "keep listening" for foreground hands-free use. Each listening window ends
  after a pause or at ten seconds; a tone indicates a new window. The microphone
  is released while synthesized speech plays. Return to the app to resume after
  switching tabs or locking the screen.
- Alt+V invokes Speak command (or finishes a recording). Escape and Stop
  everything stop locally without waiting for the speech server.
- Press Speak command to interrupt an announcement and record your command.
  This is button interruption, not simultaneous voice barge-in. Speech is not
  monitored while the app itself is talking. Use the local Stop button/Escape
  if server transcription is slow or unavailable; a spoken "stop" needs the
  network and transcription worker to finish.
- Spoken output can be unchecked to use a screen reader instead. Controls retain
  native labels, keyboard behavior and live-region messages. Screen-reader audio
  is outside the app's speech scheduler; use a headset or push-to-talk to avoid
  capturing that audio.
- The browser selects an English voice, preferring a local one. Availability and
  whether fallback voices use a platform network service depend on the device.
  There is no new Python TTS dependency.

## Audio and session behavior

Camera health alerts interrupt normal output. New direction cues replace old
ones; the app never accumulates a queue of directions. Ordinary acknowledgements
may occupy one pending slot behind an alert, expiring after eight seconds.
Directions are checked while playing and cancelled on stale observations, loss,
configuration changes, or a different direction. Current guidance repeats no
more often than every eight seconds; searching/ambiguity prompts every fifteen.
These are camera pan/centering cues. There are no walking, distance, or path-clear
claims in the new voice layer.

Audio input requests use request IDs plus session/configuration tokens. A late
response cannot overwrite a changed selection. CPU endpointing uses a simple
microphone energy threshold to stop after 1.2 seconds of silence; the existing
server Silero VAD still validates speech. Noise or a quiet microphone may make
endpointing less effective; the Finish recording button and ten-second limit
remain available. Hands-free mode can hear other people's speech: there is no
wake-word gate or speaker verification in this build.

Paused guidance continues vision tracking. A cancelled target sends
`target_active: false` to the backend and discards its appearance reference.
Camera restart/disconnection also ends that target session. Confidence, image
size and OCR configuration updates preserve the reference. Camera startup now
begins with no active target; select an object or speak a target command.

OCR is read only on request, limited to six readings / 450 characters per request.
It is described as a recent text snapshot. OCR content is displayed as text and
spoken as data; it is never passed to the command interpreter. It does not prove
which room a sign belongs to. Existing appearance matching remains a heuristic,
not a guarantee that visually identical objects are distinguishable.

## Validation status and checks

Source changes were inspected statically and the archive contents were checked.
The following tests are INCLUDED BUT NOT RUN. No runtime compatibility, real
microphone behavior, GB10 latency, or browser TTS behavior has been verified here.

Run on your existing environment after extracting, if you choose:

```bash
python -m unittest discover -s tests_voice_update -p 'test_*.py' -v
```

The Python tests cover controls/politeness/negation, side/ordinal replies, and the
WebSocket lifecycle for clearing and preserving appearance references. They use a
fake detector and no model weights. The protocol tests open a temporary local
HTTP server using aiohttp's existing testing helpers.

Optional JavaScript tests, requiring Node.js only on a development machine:

```bash
node --test tests_voice_update/speech.test.cjs
node --check yolo_live/static/app.js
node --check yolo_live/static/voice.js
node --check yolo_live/static/speech.js
```

The JavaScript tests use fake speech APIs to check expiry, priority, cancellation,
late callbacks, and microphone output blocking. They do not validate real browser
speech or replace device testing. Node is not needed by the running app.

Manual acceptance checks before relying on this build:

1. With camera off, enable voice and say the polite chair request above. Confirm
   microphone/camera permissions, target selection and spoken camera cues.
2. Put two chairs in view. Answer "the left one" after the ambiguity question.
   Let that question expire, or change the target, then verify a delayed "left
   one" asks for a complete request instead of changing an existing target.
3. Lock one object, pause, move it out of view and back, then resume. Verify the
   reference was preserved. Cancel target and verify no target guidance continues.
4. Change confidence or toggle OCR while tracking: the same reference should stay.
5. Disconnect camera/network while a direction is speaking. Verify old directions
   stop and an audible error/status replaces them. Restore camera with a command.
6. Record a command and change target manually before transcription returns. The
   old speech result must not select a new target.
7. Read text, interrupt with Speak command, then press Escape. Camera and microphone
   must turn off; late callbacks must not restart them or finish an old command.
8. Enable hands-free mode. Verify the app does not transcribe its own speech,
   silent windows are skipped when metering works, and mic permission errors stop
   automatic retries. Hide the tab: listening must stop and require resumption.
9. Check keyboard and your intended screen reader on the actual laptop/phone,
   including browser permission dialogs, denied permissions, absent microphone,
   absent TTS voices, and speech output failing partway through an announcement.

This is the voice-accessible object-finding milestone, not a deployment-grade
walking-navigation release. Depth/traversability, drop-off and overhead hazard
handling, a native app's background/audio lifecycle, speaker/wake-word handling,
and supervised accessibility/usability evaluation remain separate work.

Browser references:
- Microphone permissions and secure contexts: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- Cancelling spoken utterances: https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis/cancel
- Echo cancellation constraints: https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints/echoCancellation