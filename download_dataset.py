import tensorflow as tf
import os

data_dir = "data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print("🔽 Downloading dataset...")
tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
    fname="speech_commands_v0.02.tar.gz",
    extract=False,  # نوقف فك الضغط التلقائي
    cache_dir=".",
    cache_subdir=data_dir
)
print("✅ Done downloading.")
