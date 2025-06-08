# BeatSample Organizer

BeatSample Organizer is an advanced, Python-based command-line tool for beat creators and music producers. It scans directories for audio files, extracts metadata (such as duration and sample rate), performs BPM detection using `librosa`, and generates mel-frequency spectrogram visualizations with a customizable dark/light theme. Users can also generate a detailed JSON report of their sample library.

## Features

- **Asynchronous Scanning:**  
  Uses a ThreadPoolExecutor to efficiently scan large directories.

- **Metadata Extraction:**  
  Retrieves file information (duration, sample rate) using [mutagen](https://mutagen.readthedocs.io/en/latest/).

- **BPM Detection:**  
  Computes the beats per minute (BPM) of each audio file with [librosa](https://librosa.org/).

- **Spectrogram Generation:**  
  Generates spectrogram images for each sample with a toggle for dark or light themes using `matplotlib` and `librosa.display`.

- **JSON Reporting:**  
  Optionally outputs all extracted metadata into a JSON report for further analysis.

## Requirements

- Python 3.6 or later
- [mutagen](https://pypi.org/project/mutagen/)
- [librosa](https://pypi.org/project/librosa/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [numpy](https://pypi.org/project/numpy/)

Install the dependencies using pip:

```bash
pip install mutagen librosa matplotlib numpy
