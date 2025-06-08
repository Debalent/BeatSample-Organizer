#!/usr/bin/env python3
"""
BeatSample Organizer CLI

This tool scans a given directory for audio files, extracts metadata (duration, sample rate),
analyzes BPM using librosa, and optionally generates a spectrogram image (in dark or light theme).
It uses asynchronous processing for faster scanning of large sample libraries.
"""

import argparse
import os
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from mutagen import File as AudioFile
import librosa
import librosa.display
import matplotlib.pyplot as plt

def get_bpm(filepath):
    """
    Use librosa to load the audio file and compute its BPM.
    Returns the rounded tempo (BPM) or None if processing fails.
    """
    try:
        y, sr = librosa.load(filepath, sr=None)
        tempo, _ = librosa.beat.beat_track(y, sr=sr)
        return round(tempo)
    except Exception as e:
        print(f"Error processing BPM for {filepath}: {e}")
        return None

def generate_spectrogram(filepath, output_image, theme):
    """
    Generate and save a mel-frequency spectrogram image for the given audio file.
    The visualization uses a 'dark' theme if specified; otherwise, defaults to light mode.
    """
    try:
        y, sr = librosa.load(filepath, sr=None)
        # Set the matplotlib style based on the theme parameter.
        if theme.lower() == "dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
            
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Frequency Spectrogram')
        plt.tight_layout()
        plt.savefig(output_image)
        plt.close()
        print(f"Spectrogram saved to {output_image}")
    except Exception as e:
        print(f"Error generating spectrogram for {filepath}: {e}")

def process_file(filepath, generate_spec, theme):
    """
    Process an individual audio file:
      - Extract basic metadata (duration and sample rate) using mutagen.
      - Compute its BPM.
      - Optionally generate a spectrogram image.
    Returns a dictionary with the file's metadata.
    """
    try:
        audio = AudioFile(filepath)
        if audio is not None and audio.info:
            duration = audio.info.length
            sample_rate = getattr(audio.info, 'sample_rate', None)
            bpm = get_bpm(filepath)
            spec_path = None

            if generate_spec:
                base, _ = os.path.splitext(filepath)
                spec_path = f"{base}_spectrogram.png"
                generate_spectrogram(filepath, spec_path, theme)

            return {
                "filename": os.path.basename(filepath),
                "path": filepath,
                "duration": round(duration, 2),
                "sample_rate": sample_rate,
                "bpm": bpm,
                "spectrogram": spec_path
            }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

def scan_directory_async(directory, generate_spec, theme):
    """
    Recursively scan the provided directory for supported audio files.
    Uses asynchronous processing (ThreadPoolExecutor) for efficiency.
    Returns a list of metadata dictionaries for each processed file.
    """
    audio_extensions = ('.mp3', '.wav', '.flac', '.ogg')
    files_to_process = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(audio_extensions):
                files_to_process.append(os.path.join(root, file))

    samples = []
    if files_to_process:
        with ThreadPoolExecutor() as executor:
            # Map each file to the processing function.
            results = executor.map(lambda f: process_file(f, generate_spec, theme), files_to_process)
            for result in results:
                if result:
                    samples.append(result)
    return samples

def list_samples(samples):
    """
    Print a formatted list of processed audio samples and their metadata.
    """
    if not samples:
        print("No audio samples found.")
        return

    print("Found Audio Samples:\n")
    for idx, sample in enumerate(samples, start=1):
        sr = sample.get("sample_rate", "Unknown")
        bpm = sample.get("bpm", "N/A")
        spec = sample.get("spectrogram", "No Spectrogram")
        print(f"{idx}. {sample['filename']}")
        print(f"   Duration: {sample['duration']} s, Sample Rate: {sr}, BPM: {bpm}")
        if spec:
            print(f"   Spectrogram: {spec}")

def save_report(samples, output_file="samples_report.json"):
    """
    Save the scanned sample metadata to a JSON file.
    """
    try:
        with open(output_file, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Report saved to {output_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="BeatSample Organizer: Advanced audio analysis and organization for your beat samples."
    )
    parser.add_argument("directory", type=str, help="Directory to scan for beat samples")
    parser.add_argument("--report", action="store_true", help="Generate a JSON report of found samples")
    parser.add_argument("--spectrogram", action="store_true", help="Generate spectrogram images for each sample")
    parser.add_argument(
        "--theme",
        type=str,
        choices=["dark", "light"],
        default="light",
        help="Theme for spectrogram visualization: dark or light (default: light)"
    )

    args = parser.parse_args()

    samples = scan_directory_async(args.directory, args.spectrogram, args.theme)
    list_samples(samples)

    if args.report:
        save_report(samples)

if __name__ == "__main__":
    main()
