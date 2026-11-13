import pathlib

data_dir = pathlib.Path("data/speech_commands_v0.02")
labels = sorted([f.name for f in data_dir.glob("*") if f.is_dir() and not f.name.startswith("_")])

for i, label in enumerate(labels):
    print(f"{i}: {label}")
