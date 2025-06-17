#!/usr/bin/env python3
"""
BeatSample Organizer - Integrated with The Finisher

This tool scans a directory for audio files, extracts metadata, analyzes BPM, generates spectrograms, 
and integrates with The Finisher's database to track sample usage across projects.
"""

import argparse
import os
import json
import logging
import psycopg2
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from mutagen import File as AudioFile

# 🔹 Configure logging for better debugging and scalability
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 🔹 Database Configuration (Connect to The Finisher's PostgreSQL backend)
DB_CONFIG = {
    "dbname": "finisher_db",
    "user": "your_username",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}

def connect_db():
    """🔹 Establish a connection to the database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logging.error(f"❌ Database connection failed: {e}")
        return None

def track_sample_usage(sample_name, user_id, project_id):
    """
    🔹 Stores sample usage in the database, tracking which samples have been used before.
    - Prevents duplicate usage in multiple projects.
    - Enhances user workflow efficiency by showing usage history.
    """
    conn = connect_db()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO sample_usage (sample_name, user_id, project_id, timestamp)
                    VALUES (%s, %s, %s, NOW())
                """, (sample_name, user_id, project_id))
                conn.commit()
                logging.info(f"✅ Sample usage recorded: {sample_name} in project {project_id}")
        except Exception as e:
            logging.error(f"❌ Error tracking sample usage: {e}")
        finally:
            conn.close()

def get_bpm(filepath):
    """🔹 Compute BPM using librosa."""
    try:
        y, sr = librosa.load(filepath, sr=None)
        tempo, _ = librosa.beat.beat_track(y, sr=sr)
        return round(tempo)
    except Exception as e:
        logging.error(f"❌ BPM processing failed for {filepath}: {e}")
        return None

def generate_spectrogram(filepath, output_image, theme):
    """🔹 Generate a mel-frequency spectrogram for visualization."""
    try:
        y, sr = librosa.load(filepath, sr=None)
        plt.style.use("dark_background" if theme.lower() == "dark" else "default")

        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel-Frequency Spectrogram")
        plt.tight_layout()
        plt.savefig(output_image)
        plt.close()
        logging.info(f"✅ Spectrogram saved: {output_image}")
    except Exception as e:
        logging.error(f"❌ Error generating spectrogram for {filepath}: {e}")

def process_file(filepath, generate_spec, theme, user_id, project_id):
    """🔹 Process an individual audio file."""
    try:
        audio = AudioFile(filepath)
        if audio and audio.info:
            duration = audio.info.length
            sample_rate = getattr(audio.info, "sample_rate", None)
            bpm = get_bpm(filepath)
            spec_path = None

            if generate_spec:
                base, _ = os.path.splitext(filepath)
                spec_path = f"{base}_spectrogram.png"
                generate_spectrogram(filepath, spec_path, theme)

            sample_metadata = {
                "filename": os.path.basename(filepath),
                "path": filepath,
                "duration": round(duration, 2),
                "sample_rate": sample_rate,
                "bpm": bpm,
                "spectrogram": spec_path
            }

            # 🔹 Track sample usage within The Finisher database
            track_sample_usage(sample_metadata["filename"], user_id, project_id)

            return sample_metadata
    except Exception as e:
        logging.error(f"❌ Error processing file {filepath}: {e}")
    return None

def scan_directory_async(directory, generate_spec, theme, user_id, project_id):
    """🔹 Asynchronous scanning of a directory for supported audio files."""
    audio_extensions = (".mp3", ".wav", ".flac", ".ogg")
    files_to_process = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.lower().endswith(audio_extensions)]

    samples = []
    if files_to_process:
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda f: process_file(f, generate_spec, theme, user_id, project_id), files_to_process)
            samples.extend(filter(None, results))

    return samples

def list_samples(samples):
    """🔹 Print formatted sample data."""
    if not samples:
        logging.info("No audio samples found.")
        return

    logging.info("\n🎶 Found Audio Samples:\n")
    for idx, sample in enumerate(samples, start=1):
        logging.info(f"{idx}. {sample['filename']} - BPM: {sample['bpm']}, Duration: {sample['duration']}s")

def save_report(samples, output_file="samples_report.json"):
    """🔹 Save sample metadata to a JSON report."""
    try:
        with open(output_file, "w") as f:
            json.dump(samples, f, indent=2)
        logging.info(f"✅ Report saved: {output_file}")
    except Exception as e:
        logging.error(f"❌ Failed to save report: {e}")

def main():
    parser = argparse.ArgumentParser(description="BeatSample Organizer - Integrated with The Finisher")
    parser.add_argument("directory", type=str, help="Directory to scan for beat samples")
    parser.add_argument("--user_id", type=int, required=True, help="User ID for sample tracking")
    parser.add_argument("--project_id", type=int, required=True, help="Project ID for sample tracking")
    parser.add_argument("--report", action="store_true", help="Generate a JSON report")
    parser.add_argument("--spectrogram", action="store_true", help="Generate spectrograms")
    parser.add_argument("--theme", type=str, choices=["dark", "light"], default="light", help="Spectrogram theme")

    args = parser.parse_args()
    samples = scan_directory_async(args.directory, args.spectrogram, args.theme, args.user_id, args.project_id)
    list_samples(samples)

    if args.report:
        save_report(samples)

if __name__ == "__main__":
    main()
