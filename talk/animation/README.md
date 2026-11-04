# Timothy explainer animation (talk)

Goofy doodle animation for the "how GR00T works / why Timothy didn't" part of the talk.
Output: `timothy_explainer.mp4` (1920x1080, 30 fps, 2:28, your voice as audio).

Files you provide (not in git):
- `voice.m4a` - the voice recording
- `timothy.png` - the cut-out photo of the arm (from Downloads/robot_arm_cutout.png)

Rebuild:
    uv run python transcribe.py               # voice.m4a -> words.json (word timings; needs voice16k.wav, see below)
    uv run python anim.py                     # whole video (~1 min on the laptop)
    uv run python anim.py --from 40 --to 52   # just a piece, with matching audio
    uv run python anim.py --stills 12,30      # PNG frames in stills/ (sheet.py makes a contact sheet)

voice16k.wav: `ffmpeg -i voice.m4a -ac 1 -ar 16000 voice16k.wav` (ffmpeg comes with the
imageio-ffmpeg package: `uv run python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"`).

Scenes and timings are the SCENES list at the bottom of anim.py; times come from words.json.
If you re-record, rerun transcribe.py and shift the times.

Credits / notes:
- Fonts: Ink Free and Comic Sans MS (Microsoft, come with Windows; not included here).
- Speech-to-text: faster-whisper (MIT) with OpenAI's Whisper small.en model (MIT).
- The ChatGPT, Isaac GR00T, Cosmos Reason and Claude characters are hand-made parody doodles,
  not the companies' logos. ChatGPT is OpenAI's, GR00T/Cosmos are NVIDIA's, Claude is Anthropic's.
