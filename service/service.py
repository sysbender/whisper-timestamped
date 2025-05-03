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

# Language to spaCy model mapping
LANG_TO_SPACY = {
    'en': 'en_core_web_trf',
    'fr': 'fr_core_news_sm',
    # Add more languages as needed
}

# Global model cache
_model = None

def get_model():
    """Get or initialize the Whisper model"""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = whisper_timestamped.load_model("medium", device=device)
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
    """Create necessary directories if they don't exist, gracefully handle mounted volumes"""
    try:
        # First verify /data exists since it should be mounted
        if not DATA_DIR.exists():
            raise RuntimeError(f"Data directory {DATA_DIR} does not exist. Please make sure the volume is mounted correctly.")
        
        # Create subdirectories if they don't exist
        for directory in [INPUT_DIR, PROCESSING_DIR, OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Verify write permissions
        for directory in [INPUT_DIR, PROCESSING_DIR, OUTPUT_DIR]:
            test_file = directory / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
            except (OSError, PermissionError) as e:
                raise RuntimeError(f"No write permission in {directory}. Error: {str(e)}")
                
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

        # Get cached model and transcribe
        model = get_model()
        result = whisper_timestamped.transcribe(model, str(audio_file))

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

        # Cleanup processing directory
        os.remove(audio_file)
        os.remove(json_path)
        logging.info(f"Successfully processed {input_file.name}")

    except Exception as e:
        logging.error(f"Error processing {input_file.name}: {str(e)}")
        # Move problematic file back to input directory
        if proc_file.exists():
            shutil.move(str(proc_file), str(input_file))

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