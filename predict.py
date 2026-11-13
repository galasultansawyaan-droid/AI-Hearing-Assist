import tensorflow as tf
import librosa
import numpy as np
import pathlib

model = tf.keras.models.load_model("sound_model.h5")

file_path = "test_recording.wav"
audio, sr = librosa.load(file_path, sr=16000)

audio = audio[:16000]
audio = np.expand_dims(audio, axis=0)

pred = model.predict(audio)[0]
top_indices = pred.argsort()[-3:][::-1]

data_dir = pathlib.Path("data/speech_commands_v0.02")
labels = sorted([f.name for f in data_dir.glob("*") if f.is_dir() and not f.name.startswith("_")])

print("🔊 Top 3 Words Predictions:")
for i in top_indices:
    label = labels[i] if i < len(labels) else "Unknown"
    confidence = pred[i]
    print(f"{label}: {confidence:.4f}")

def recognize_speaker(audio):
    speaker_model = tf.keras.models.load_model("speaker_model.h5")
    audio_processed = np.expand_dims(audio, axis=0)
    speaker_pred = speaker_model.predict(audio_processed)[0]
    
    speakers = ['Abeer Alqarny', 'Hala Abdullah', 'Gala Sawyaan']
    
    predicted_speaker_index = np.argmax(speaker_pred)
    predicted_speaker = speakers[predicted_speaker_index]
    speaker_confidence = speaker_pred[predicted_speaker_index]
    
    return predicted_speaker, speaker_confidence

predicted_speaker, speaker_confidence = recognize_speaker(audio)

print(f"\nPredicted Speaker: {predicted_speaker} with confidence: {speaker_confidence:.4f}")
