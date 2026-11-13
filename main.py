import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
audio_path = "test_recording.wav"
y, sr = librosa.load(audio_path)

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Display MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
