# AIHearingAssist 🎧

A small speech-recognition project that listens to a short voice recording and predicts which spoken word it contains. It is trained on Google's **Speech Commands v0.02** dataset using a 1D Convolutional Neural Network (CNN) built with TensorFlow / Keras.

## Features

- Download and extract the Speech Commands dataset automatically
- Train a 1D CNN on 1-second, 16 kHz audio clips
- Record your own voice from the microphone
- Predict the **top 3 most likely words** with confidence scores
- Visualize MFCC features of a recording

## Project Structure

| File | Purpose |
|------|---------|
| `download_dataset.py` | Downloads the Speech Commands v0.02 dataset |
| `extract_dataset.py` | Extracts the downloaded `.tar.gz` into `data/` |
| `get_labels.py` | Prints the list of word labels found in the dataset |
| `train_model.py` | Loads and preprocesses the dataset (train/validation split) |
| `cnn_model.py` | Defines, trains and saves the CNN model (`sound_model.h5`) |
| `record.py` | Records 10 seconds from your microphone to `test_recording.wav` |
| `predict.py` | Loads the model and predicts the word in `test_recording.wav` |
| `main.py` | Plots the MFCC features of `test_recording.wav` |
| `sound_model.h5` | Pre-trained model |

## Model

- Input: 16,000 audio samples (1 second at 16 kHz), normalized to [-1, 1]
- Layers: Reshape → Conv1D(16) → MaxPool → Conv1D(32) → MaxPool → Flatten → Dense(128) → Dense(softmax)
- Optimizer: Adam, loss: sparse categorical cross-entropy, 5 epochs

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/AIHearingAssist.git
cd AIHearingAssist
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

```bash
python download_dataset.py
python extract_dataset.py
```

### 3. Train the model (optional, a trained `sound_model.h5` is included)

```bash
python cnn_model.py
```

### 4. Record and predict

```bash
python record.py      # speak a word clearly
python predict.py     # prints the top 3 predicted words
python main.py        # optional: show MFCC plot
```

## Tech Stack

Python 3.12 · TensorFlow / Keras · librosa · NumPy · Matplotlib · sounddevice · soundfile

## Dataset

[Speech Commands v0.02](https://www.tensorflow.org/datasets/catalog/speech_commands) by Google (Creative Commons BY 4.0). The dataset is **not** included in this repository because of its size.

## Notes

- `predict.py` also contains an optional speaker-recognition step that needs a separate `speaker_model.h5`, which is not included in this repo.

## Author

Gala Sultan Sawyaan · [GitHub](https://github.com/galasultansawyaan-droid)

## License

This project is for educational purposes.
