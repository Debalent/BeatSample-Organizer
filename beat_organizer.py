#!/usr/bin/env python3
import argparse
import os
import json
from mutagen import File as AudioFile

def scan_directory(directory):
    """
    Scan the given directory for audio files and return a list of file data.
    The file data includes file name, duration and sometimes sample rate (if available).
    """
    audio_extensions = ('.mp3', '.wav', '.flac', '.ogg')
    samples = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(audio_extensions):
                filepath = os.path.join(root, file)
                try:
                    audio = AudioFile(filepath)
                    if audio is not None and audio.info:
                        duration = audio.info.length  # duration in seconds
                        sample_rate = getattr(audio.info, 'sample_rate', None)
                        samples.append({
                            'filename': file,
                            'path': filepath,
                            'duration': round(duration, 2),
                            'sample_rate': sample_rate
                        })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    return samples

def list_samples(samples):
    """Print a list of beat samples with their details."""
    if not samples:
        print("No audio samples found.")
        return

    print("Found Audio Samples:")
    for idx, sample in enumerate(samples, start=1):
        sr = sample.get('sample_rate', 'Unknown')
        print(f"{idx}. {sample['filename']} - Duration: {sample['duration']}s, Sample Rate: {sr}")

def save_report(samples, output_file="samples_report.json"):
    """Save the samples' data to a JSON file for further analysis."""
    try:
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"Report saved to {output_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="BeatSample Organizer: Analyze and organize your beat samples."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory to scan for beat samples"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a JSON report of found samples"
    )

    args = parser.parse_args()
    samples = scan_directory(args.directory)
    
    list_samples(samples)
    if args.report:
        save_report(samples)

if __name__ == "__main__":
    main()
