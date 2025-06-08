# BeatSample Organizer

BeatSample Organizer is a Python-based command-line tool designed for beat creators and music producers. It scans directories for audio files (such as WAV, MP3, FLAC, and OGG), extracts essential metadata like duration and sample rate, lists your beat samples, and can generate a JSON report for further analysis. This tool helps you keep your sample library organized with ease.

## Features

- **Scan Samples:** Recursively search a specified directory for audio files.
- **Metadata Extraction:** Uses the [mutagen](https://mutagen.readthedocs.io/en/latest/) library to extract details such as duration and sample rate.
- **List Samples:** Display your beat sample files along with relevant metadata.
- **Generate Report:** Optionally output the scanned data into a JSON report for further analysis.

## Requirements

- Python 3.6 or higher
- [Mutagen](https://mutagen.readthedocs.io/en/latest/) library

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/BeatSample-Organizer.git
   cd BeatSample-Organizer
