import tensorflow as tf
import os
import pathlib

data_dir = pathlib.Path("data/speech_commands_v0.02")
commands = tf.io.gfile.listdir(str(data_dir))
commands = [cmd for cmd in commands if os.path.isdir(data_dir/cmd) and cmd != '_background_noise_']

print(f"Labels: {commands}")


AUTOTUNE = tf.data.AUTOTUNE
batch_size = 64
seed = 42
output_dim = len(commands)

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

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

print("✅ Dataset ready for model training")
