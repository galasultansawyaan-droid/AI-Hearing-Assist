import tarfile

tar_path = "data/speech_commands_v0.02.tar.gz"

with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path="data")

print("✅ Extraction completed.")
