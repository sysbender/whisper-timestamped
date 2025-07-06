#!/usr/bin/env python3

import os
import time
import shutil
import json
import logging
import subprocess
from pathlib import Path
import whisper_timestamped
import torch
import webvtt
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure paths using environment variables with /data base dir
DATA_DIR = Path(os.getenv('DATA_DIR', '/data'))
INPUT_DIR = DATA_DIR / 'input'
PROCESSING_DIR = DATA_DIR / 'processing'
OUTPUT_DIR = DATA_DIR / 'output'
ARCHIVE_DIR = DATA_DIR / 'archive'
MODELS_DIR = DATA_DIR / 'models'

# Model configuration
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'large-v3')

# Language to spaCy model mapping
LANG_TO_SPACY = {
    'en': 'en_core_web_trf',
    'fr': 'fr_core_news_sm',
    # Add more languages as needed
}

# Global model cache
_model = None

def get_model():
    """Get or initialize the Whisper model with local caching"""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Configure whisper to use our models directory
        os.environ["WHISPER_MODELS_DIR"] = str(MODELS_DIR)
        
        # Check if model already exists in cache
        model_path = MODELS_DIR / f"{WHISPER_MODEL}.pt"
        if not model_path.exists():
            logging.info(f"Downloading model {WHISPER_MODEL} to {MODELS_DIR}...")
        
        _model = whisper_timestamped.load_model(WHISPER_MODEL, device=device, download_root=str(MODELS_DIR))
        logging.info(f"Model {WHISPER_MODEL} loaded on {device}")
    return _model

def extract_audio_from_mp4(mp4_path: Path) -> Path:
    """Extract audio from MP4 file using ffmpeg"""
    output_path = mp4_path.with_suffix('.mp3')
    cmd = [
        'ffmpeg', '-i', str(mp4_path),
        '-q:a', '0', '-map', 'a', str(output_path),
        '-y'  # Overwrite output files
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path

def ensure_directories():
    """Ensure all required directories exist and have write permissions."""
    try:
        # First verify /data exists since it should be mounted
        if not DATA_DIR.exists():
            raise RuntimeError(f"Data directory {DATA_DIR} does not exist. Please make sure the volume is mounted correctly.")
        
        # Create subdirectories if they don't exist
        for directory in [INPUT_DIR, PROCESSING_DIR, OUTPUT_DIR, ARCHIVE_DIR, MODELS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            # Verify write permissions
            if not os.access(directory, os.W_OK):
                raise PermissionError(f"No write permission for directory: {directory}")
                
        logging.info("Directory structure verified and ready")
        
    except Exception as e:
        logging.error(f"Failed to setup directory structure: {str(e)}")
        raise

def process_file(input_file: Path):
    """Process a single audio file"""
    try:
        # Move to processing directory
        proc_file = PROCESSING_DIR / input_file.name
        shutil.move(str(input_file), str(proc_file))
        logging.info(f"Processing {proc_file}")

        # Handle MP4 files
        if proc_file.suffix.lower() == '.mp4':
            audio_file = extract_audio_from_mp4(proc_file)
            os.remove(proc_file)  # Remove original MP4 after extraction
        else:
            audio_file = proc_file

        # Get cached model and transcribe with accurate settings
        model = get_model()
        result = whisper_timestamped.transcribe(
            model, 
            str(audio_file), 
            vad='silero',  # Use silero VAD
            beam_size=5,   # Enable beam search
            best_of=5,     # Compare multiple candidates
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # Try different sampling temperatures
        )

        # Save timestamped JSON
        json_path = audio_file.with_suffix('.words.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Get detected language and map to spaCy model
        detected_lang = result.get('language', 'en')
        spacy_model = LANG_TO_SPACY.get(detected_lang, 'en_core_web_trf')
        logging.info(f"Detected language: {detected_lang}, using spaCy model: {spacy_model}")

        # Generate VTT using segmentation tool with punctuation-only mode
        vtt_path = OUTPUT_DIR / audio_file.with_suffix('.vtt').name
        cmd = [
            'python3',
            '/usr/src/app/segmentation/segment.py',  # Use absolute path in Docker
            '--format', 'vtt',
            '--lang', detected_lang,
            '--output', str(vtt_path),
            str(json_path)
        ]
        with open(vtt_path, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f)

        # Archive processed files and create adjusted VTT
        archive_audio_path = ARCHIVE_DIR / audio_file.name
        archive_json_path = ARCHIVE_DIR / json_path.name
        archive_vtt_path = ARCHIVE_DIR / vtt_path.name
        
        # Create adjusted VTT with buffered end times
        adjusted_vtt_path = ARCHIVE_DIR / f"{vtt_path.stem}_EndTimeAdjusted.vtt"
        shift_vtt_times(vtt_path, adjusted_vtt_path)

        # Move and copy files to appropriate locations
        shutil.move(str(audio_file), str(archive_audio_path))
        shutil.move(str(json_path), str(archive_json_path))
        shutil.copy2(str(vtt_path), str(archive_vtt_path))  # Copy original VTT to archive
        shutil.copy2(str(adjusted_vtt_path), str(vtt_path))  # Replace output VTT with adjusted version

        logging.info(f"Successfully processed and archived {input_file.name}")

    except Exception as e:
        logging.error(f"Error processing {input_file.name}: {str(e)}")
        # Move problematic file back to input directory
        if proc_file.exists():
            shutil.move(str(proc_file), str(input_file))

import webvtt
from datetime import datetime, timedelta

def shift_time_string(time_str: str, buffer_seconds: float) -> str:
    """Shift a VTT time string (e.g., '00:00:01.000') by buffer_seconds forward."""
    # Parse the time string
    hours, minutes, seconds = time_str.split(":")
    sec, ms = seconds.split(".")
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(sec) + float(f"0.{ms}")
    new_total = total_seconds + buffer_seconds
    h = int(new_total // 3600)
    m = int((new_total % 3600) // 60)
    s = new_total % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def shift_vtt_times(input_path: Path, output_path: Path, buffer_seconds: float = 0.2):
    """Shift both start and end times of all VTT subtitle lines by buffer_seconds forward.
    
    Args:
        input_path (Path): Path to input VTT file
        output_path (Path): Path to output VTT file
        buffer_seconds (float): Seconds to shift each segment's start and end time (default: 0.2)
    """
    # Read the VTT file
    vtt = webvtt.read(str(input_path))
    
    # Process each caption
    for caption in vtt.captions:
        caption.start = shift_time_string(caption.start, buffer_seconds)
        caption.end = shift_time_string(caption.end, buffer_seconds)
    
    # Save the modified VTT
    vtt.save(str(output_path))

def service_loop():
    """Main service loop that monitors the input directory"""
    logging.info("Starting transcription service...")
    
    # Verify directory structure on startup
    ensure_directories()

    while True:
        try:
            # Get all MP3 and MP4 files from input directory
            input_files = sorted(
                list(INPUT_DIR.glob('*.mp3')) + list(INPUT_DIR.glob('*.mp4')),
                key=lambda p: p.stat().st_mtime  # Process oldest files first
            )
            
            if input_files:
                # Process one file at a time
                process_file(input_files[0])
            else:
                time.sleep(2)  # Reduced sleep time for more responsiveness
                
        except Exception as e:
            logging.error(f"Service error: {str(e)}")
            time.sleep(2)  # Wait before retrying

if __name__ == "__main__":
    service_loop()