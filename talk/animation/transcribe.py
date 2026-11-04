"""Word timestamps for voice.m4a -> words.json (faster-whisper, CPU)."""
import json, wave
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("small.en", device="cpu", compute_type="int8")
w = wave.open("voice16k.wav")
audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
segs, info = model.transcribe(audio, word_timestamps=True)
words = []
for s in segs:
    for w in s.words:
        words.append({"w": w.word.strip(), "t0": round(w.start, 2), "t1": round(w.end, 2)})
json.dump({"duration": info.duration, "words": words}, open("words.json", "w"), indent=0)
print(f"duration {info.duration:.1f} s, {len(words)} words")
