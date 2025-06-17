#!/usr/bin/env python3
"""
BeatSample Organizer - Now DAW-Compatible!

Enhanced Features:
- Auto-detect DAW project files & extract sample metadata
- Analyze key, BPM, duration, sample rate for deeper insights
- Organize samples intelligently based on user preferences
"""

import argparse
import os
import json
import logging
import psycopg2
import librosa
import numpy as np
import re
from mutagen import File as AudioFile
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import matplotlib.pyplot as plt

# 🔹 Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 🔹 DAW Project File Extensions
DAW_EXTENSIONS = {
    "Ableton": ".als",
    "FL Studio": ".flp",
    "Logic Pro": ".logicx",
    "Pro Tools": ".ptx"
}

# 🔹 Supported Audio Formats
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".aiff")

# 🔹 Database Configuration
DB_CONFIG = {
    "dbname": "finisher_db",
    "user": "your_username",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}

# 🔹 Connect to The Finisher's Database
def connect_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logging.error(f"❌ Database connection failed: {e}")
        return None

# 🔹 Track Sample Usage
def track_sample_usage(sample_name, user_id, project_id):
    conn = connect_db()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO sample_usage (sample_name, user_id, project_id, timestamp)
                    VALUES (%s, %s, %s, NOW())
                """, (sample_name, user_id, project_id))
                conn.commit()
                logging.info(f"✅ Tracked: {sample_name} in project {project_id}")
        except Exception as e:
            logging.error(f"❌ Error tracking usage: {e}")
        finally:
            conn.close()

# 🔹 Extract BPM
def get_bpm(filepath):
    try:
        y, sr = librosa.load(filepath, sr=None)
        tempo, _ = librosa.beat.beat_track(y, sr=sr)
        return round(tempo)
    except Exception as e:
        logging.error(f"❌ BPM analysis failed for {filepath}: {e}")
        return None

# 🔹 Detect Key
def get_key(filepath):
    try:
        y, sr = librosa.load(filepath, sr=None)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        avg_chroma = np.mean(chroma, axis=1)
        key_index = np.argmax(avg_chroma)
        key_mapping = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return key_mapping[key_index]
    except Exception as e:
        logging.error(f"❌ Key detection failed for {filepath}: {e}")
        return None

# 🔹 Generate Spectrograms
def generate_spectrogram(filepath, output_image, theme):
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
        logging.error(f"❌ Error generating spectrogram: {e}")

# 🔹 Process Audio Files
def process_file(filepath, generate_spec, theme, user_id, project_id):
    try:
        audio = AudioFile(filepath)
        if audio and audio.info:
            duration = round(audio.info.length, 2)
            sample_rate = getattr(audio.info, "sample_rate", None)
            bpm = get_bpm(filepath)
            key = get_key(filepath)
            spec_path = None

            if generate_spec:
                base, _ = os.path.splitext(filepath)
                spec_path = f"{base}_spectrogram.png"
                generate_spectrogram(filepath, spec_path, theme)

            sample_metadata = {
                "filename": os.path.basename(filepath),
                "path": filepath,
                "duration": duration,
                "sample_rate": sample_rate,
                "bpm": bpm,
                "key": key,
                "spectrogram": spec_path
            }

            # 🔹 Track sample usage in The Finisher
            track_sample_usage(sample_metadata["filename"], user_id, project_id)

            return sample_metadata
    except Exception as e:
        logging.error(f"❌ Error processing file: {e}")
    return None

# 🔹 Scan DAW Project Files
def scan_daw_files(directory):
    daw_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in DAW_EXTENSIONS.values()):
                daw_files.append(os.path.join(root, file))
    return daw_files

# 🔹 Scan Audio Samples Asynchronously
def scan_directory_async(directory, generate_spec, theme, user_id, project_id):
    files_to_process = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.lower().endswith(AUDIO_EXTENSIONS)]
    daw_projects = scan_daw_files(directory)

    samples = []
    if files_to_process:
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda f: process_file(f, generate_spec, theme, user_id, project_id), files_to_process)
            samples.extend(filter(None, results))

    logging.info(f"✅ DAW Projects Found: {daw_projects}")
    return samples

# 🔹 Command Line Interface
def main():
    parser = argparse.ArgumentParser(description="BeatSample Organizer - DAW-Compatible")
    parser.add_argument("directory", type=str, help="Directory to scan")
    parser.add_argument("--user_id", type=int, required=True, help="User ID")
    parser.add_argument("--project_id", type=int, required=True, help="Project ID")
    parser.add_argument("--report", action="store_true", help="Generate a JSON report")
    parser.add_argument("--spectrogram", action="store_true", help="Generate spectrograms")
    parser.add_argument("--theme", type=str, choices=["dark", "light"], default="light")

    args = parser.parse_args()
    samples = scan_directory_async(args.directory, args.spectrogram, args.theme, args.user_id, args.project_id)

    if args.report:
        with open("samples_report.json", "w") as f:
            json.dump(samples, f, indent=2)
        logging.info("✅ Report saved!")

if __name__ == "__main__":
    main()
