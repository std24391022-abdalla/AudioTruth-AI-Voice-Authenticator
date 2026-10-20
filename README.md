# AudioTruth: AI Voice Authenticator

**AudioTruth** is a local, GPU-backed research application that classifies an uploaded speech recording as either **Human Speech** or **AI-Generated Speech** using a trained 2D Convolutional Neural Network (CNN).

**Developer:** Abdalla  
**Target environment:** Windows, Python 3.12, NVIDIA GPU with a compatible CUDA-enabled PyTorch build.

> The application is an experimental classifier. Its score is a model output, not a calibrated real-world probability or standalone forensic proof.

## Features

- Gradio web interface served locally at `http://127.0.0.1:7860`.
- GPU-required inference: startup validates CUDA and performs a model smoke test. The application does not silently switch the inference model to CPU.
- Audio upload for WAV and MP3 recordings supported by the installed audio decoder.
- Audio preparation: convert to mono, resample to 16,000 Hz, then use a fixed 3-second segment.
- Internal 64-bin Mel-Spectrogram feature extraction. The plot itself is not shown in the presentation UI.
- Prediction card and model scores for Human Speech and AI-Generated Speech.
- Technical information and limitations are kept in a separate, initially collapsed section.
- Local-only launch (`share=False`); no public Gradio share link is created.

## Repository files

```text
AudioTruth-AI-Voice-Authenticator/
├── app.py
├── audio_deepfake_model.pth
├── requirements.txt
├── README.md
└── .gitignore
```

**Important:** `audio_deepfake_model.pth` is required at runtime and must be placed beside `app.py`. It is not included in this starter ZIP because the trained checkpoint is present on the developer's laptop, not in this generated package. Copy the existing checkpoint into the repository folder before running the app and before uploading the intended repository files.

## Requirements

- 64-bit Windows.
- Python 3.12 is the version used for the local project setup.
- A supported NVIDIA GPU and a compatible NVIDIA driver.
- Internet access during dependency installation. Once the dependencies and checkpoint are installed locally, inference is designed to run locally without downloading a model at runtime.

The tested local setup reported an NVIDIA GeForce RTX 5060 Laptop GPU, PyTorch `2.14.1+cu130`, TorchAudio `2.11.0+cu130`, Gradio `6.30.0`, NumPy `2.5.4`, and SoundFile `0.14.0`.

## Installation (Windows / CMD)

Open CMD in the project folder and run:

```bat
py -3.12 -m pip install -r requirements.txt
```

PyTorch and TorchAudio wheels are selected from the official CUDA 13.0 PyTorch wheel index specified in `requirements.txt`. The installation may download a large PyTorch package.

Check CUDA availability:

```bat
py -3.12 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')"
```

The expected result on a correctly configured GPU machine is `CUDA available: True` and the NVIDIA GPU name.

## Run the application

Ensure both `app.py` and `audio_deepfake_model.pth` are in the same folder, then run:

```bat
py -3.12 app.py
```

Open `http://127.0.0.1:7860` in a browser if it does not open automatically. Keep the CMD window open while using the application; closing it stops the server.

At startup, the application loads the checkpoint strictly and tests the feature-extraction plus CNN path. If CUDA or the checkpoint validation fails, startup stops and reports an error rather than pretending that GPU inference is active.

## Model and audio pipeline

### CNN architecture

The supplied model uses four 2D convolution blocks with 32, 64, 128, and 256 channels, Batch Normalization, ReLU, and pooling, followed by adaptive average pooling and a two-class linear classifier. The checkpoint expects class index `0` to mean **Human / Real** and class index `1` to mean **AI-Generated / Fake**.

### Preprocessing

1. Read the uploaded audio file.
2. Convert multiple channels to mono.
3. Resample to 16,000 Hz when required.
4. Pad or trim the waveform to 3 seconds (48,000 samples).
5. Compute a 64-bin Mel-Spectrogram and convert it to a decibel representation.
6. Send the feature tensor to the CUDA device for CNN inference.

The current model was set up for fixed-length 3-second training inputs. For recordings longer than 3 seconds, **only the first 3 seconds are classified**. Shorter valid recordings are padded with silence. Consequently, the displayed classification does not necessarily describe every part of a long recording. Whole-recording analysis would require a separate, tested sliding-window or segment-aggregation design.

## Training summary (reported project notes)

The project handoff notes report the following training workflow:

- A downloaded WaveFake-related archive of about 26.93 GB and a local inventory of 134,266 audio files.
- A balanced sample of 12,000 clips: 6,000 real LJSpeech clips and 6,000 generated clips from multiple vocoders.
- A split of 9,600 training clips and 2,400 validation clips (80/20).
- Five training epochs using AdamW and a ReduceLROnPlateau learning-rate scheduler.
- The notes report 100% validation accuracy and a validation loss displayed as `0.0000`.

These numbers are **reported project notes**, not an independently reproduced evaluation in this repository. A perfect validation result does not by itself establish generalization to new speakers, recording conditions, or modern unseen generators. Before making a research-performance claim, evaluate on a separate test set, verify there is no speaker/source overlap or data leakage, report a confusion matrix, precision, recall, F1, and test-set composition, and consider calibration of scores.

## Dataset reference

The WaveFake dataset was introduced by Joel Frank and Lea Schönherr. The original dataset record describes generated audio examples from several neural vocoders and lists the dataset under **CC BY-SA 4.0**. Consult the dataset record and its license for attribution and reuse requirements:

- Dataset: https://zenodo.org/records/5270336
- DOI: https://doi.org/10.5281/zenodo.5270336
- Project repository: https://github.com/RUB-SysSec/WaveFake

The dataset archive and raw audio are not included in this repository. The real LJSpeech reference data and other source materials may have their own applicable terms; review each dataset's terms before redistributing data or derived artifacts.

## Interpreting scores

Softmax scores indicate how the current model distributes its output across the two trained classes. They are **not necessarily calibrated probabilities**. A high score does not guarantee correctness, and this research prototype should not be used as the sole basis for identity verification, fraud accusations, or legal/forensic conclusions.

## Privacy and network behavior

The interface binds to `127.0.0.1` and launches with Gradio sharing disabled. This is intended for local use on the same computer. Do not expose the service to a public network without a separate security review.

## Troubleshooting

- **`GPU REQUIRED BUT NOT AVAILABLE`:** confirm the NVIDIA driver is installed, run `nvidia-smi`, and verify `torch.cuda.is_available()` is `True`.
- **Checkpoint not found:** place `audio_deepfake_model.pth` beside `app.py`.
- **Checkpoint/model architecture mismatch:** do not replace the checkpoint with random weights. Confirm that the checkpoint belongs to this CNN architecture.
- **Audio read error:** try a valid WAV file or convert the recording to PCM WAV, then retry.
- **Port 7860 is already in use:** stop the other AudioTruth process or close the application that is using the port before restarting.

## Acknowledgement

Built as an educational/research prototype by **Abdalla**. Please choose and add an appropriate software license before encouraging reuse or redistribution of this application's code. The dataset's license is separate from the software license.
