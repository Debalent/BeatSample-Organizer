#!/usr/bin/env python3
"""
BeatSample Organizer - DAW-Compatible for The Finisher

Enhanced Features:
- Auto-detect DAW project files & extract sample metadata (BPM, key, duration)
- Organize samples with database integration for The Finisher
- API endpoint for sample organization
- Scalable with connection pooling and robust error handling
"""
import argparse
import os
import json
import logging
import psycopg2
from psycopg2.extras import DictCursor
import librosa
import numpy as np
from mutagen import File as AudioFile
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.db.database import get_db

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

# 🔹 FastAPI Router
router = APIRouter()

# 🔹 Pydantic Model for API Input
class SampleOrganizeInput(BaseModel):
    directory: str
    user_id: int
    project_id: int
    generate_spectrogram: bool = False
    theme: str = "light"

# 🔹 Extract BPM
def get_bpm(filepath):
    try:
        y, sr = librosa.load(filepath, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
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
        S = librosa.feature.melspectrogram(y=y, sr=sr)
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

# 🔹 Get or Create Sample in Database
def get_or_create_sample(db, filename, path, duration, sample_rate, bpm, key, spectrogram_path):
    try:
        with db.cursor() as cursor:
            cursor.execute("""
                SELECT id FROM samples WHERE path = %s
            """, (path,))
            result = cursor.fetchone()
            if result:
                return result['id']
            cursor.execute("""
                INSERT INTO samples (filename, path, duration, sample_rate, bpm, key, spectrogram_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (filename, path, duration, sample_rate, bpm, key, spectrogram_path))
            return cursor.fetchone()['id']
    except Exception as e:
        logging.error(f"❌ Error getting/creating sample: {e}")
        raise

# 🔹 Track Sample Usage
def track_sample_usage(db, sample_id, user_id, project_id):
    try:
        with db.cursor() as cursor:
            cursor.execute("""
                INSERT INTO sample_usage (sample_id, user_id, project_id, timestamp)
                VALUES (%s, %s, %s, NOW())
            """, (sample_id, user_id, project_id))
            db.commit()
            logging.info(f"✅ Tracked: sample_id {sample_id} in project {project_id}")
    except Exception as e:
        logging.error(f"❌ Error tracking usage: {e}")
        raise

# 🔹 Process Audio Files
def process_file(filepath, generate_spec, theme, user_id, project_id, db):
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

            sample_id = get_or_create_sample(db, os.path.basename(filepath), filepath, duration, sample_rate, bpm, key, spec_path)
            track_sample_usage(db, sample_id, user_id, project_id)

            return {
                "sample_id": sample_id,
                "filename": os.path.basename(filepath),
                "path": filepath,
                "duration": duration,
                "sample_rate": sample_rate,
                "bpm": bpm,
                "key": key,
                "spectrogram": spec_path
            }
    except Exception as e:
        logging.error(f"❌ Error processing file {filepath}: {e}")
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
def scan_directory_async(directory, generate_spec, theme, user_id, project_id, db):
    files_to_process = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.lower().endswith(AUDIO_EXTENSIONS)]
    daw_projects = scan_daw_files(directory)

    samples = []
    if files_to_process:
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda f: process_file(f, generate_spec, theme, user_id, project_id, db), files_to_process)
            samples.extend(filter(None, results))

    logging.info(f"✅ DAW Projects Found: {daw_projects}")
    return samples

# 🔹 FastAPI Endpoint
@router.post("/organize-samples")
def organize_samples(input: SampleOrganizeInput, db=Depends(get_db)):
    """
    🔹 Organizes audio samples in a directory and stores metadata in the database.
    - Scans directory for audio files and DAW projects.
    - Extracts metadata (BPM, key, duration) and optionally generates spectrograms.
    - Returns processed sample metadata.
    """
    try:
        samples = scan_directory_async(
            input.directory,
            input.generate_spectrogram,
            input.theme,
            input.user_id,
            input.project_id,
            db
        )
        return {"samples": samples, "message": f"Processed {len(samples)} samples"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error organizing samples: {str(e)}")

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
    conn = connect_db()
    if not conn:
        logging.error("❌ Failed to connect to database")
        return
    try:
        samples = scan_directory_async(args.directory, args.spectrogram, args.theme, args.user_id, args.project_id, conn)
        if args.report:
            with open("samples_report.json", "w") as f:
                json.dump(samples, f, indent=2)
            logging.info("✅ Report saved!")
    finally:
        conn.close()

if __name__ == "__main__":
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router, prefix="/api")
    main()
