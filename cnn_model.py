import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib

data_dir = pathlib.Path("data/speech_commands_v0.02")

commands = tf.io.gfile.listdir(str(data_dir))
commands = [cmd for cmd in commands if cmd != "_background_noise_"]
print(f"Labels: {commands}")

batch_size = 64
seed = 42

train_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=str(data_dir),
    batch_size=batch_size,
    validation_split=0.2,
    seed=seed,
    output_sequence_length=16000,
    subset="training"
)

val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=str(data_dir),
    batch_size=batch_size,
    validation_split=0.2,
    seed=seed,
    output_sequence_length=16000,
    subset="validation"
)

def preprocess(audio, label):
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32) / 32768.0
    return audio, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(16000,)),
    layers.Reshape((1000, 16)),
    layers.Conv1D(16, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(commands), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=5)

model.save("sound_model.h5")
print("✅ Model saved as sound_model.h5")
