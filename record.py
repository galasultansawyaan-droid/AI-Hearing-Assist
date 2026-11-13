import sounddevice as sd
import soundfile as sf
import time

filename = "test_recording.wav"
duration = 10  # seconds
fs = 16000

print("🎙️ Speak now (10 seconds)...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
time.sleep(duration)
sd.wait()
sf.write(filename, recording, fs)
print("✅ Recording saved as", filename)





