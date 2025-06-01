import tensorflow as tf
import numpy as np
import librosa
import os
import csv
import sys
from pydub import AudioSegment
import tensorflow_hub as hub
from pathlib import Path

# ------------------------ Command-Line Argument ------------------------
if len(sys.argv) < 2:
    print("Usage: python detect_voice_segments.py <audio_file> [min_duration]")
    sys.exit(1)

AUDIO_FILE = sys.argv[1]
# Add command-line parameter for minimum voice segment duration (default: 2 seconds)
MIN_VOICE_DURATION = 2.0  # Default value in seconds
if len(sys.argv) >= 3:
    try:
        MIN_VOICE_DURATION = float(sys.argv[2])
        print(f"Using minimum voice segment duration: {MIN_VOICE_DURATION} seconds")
    except ValueError:
        print(f"Invalid minimum duration value. Using default: {MIN_VOICE_DURATION} seconds")

basename = Path(AUDIO_FILE).stem
OUTPUT_DIR = f"{basename}_segments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------ Constants ------------------------
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
SCORE_THRESHOLD = 0.5
SAMPLE_RATE = 16000

# ------------------------ Load Audio ------------------------
print("Loading audio...")
waveform, sr = librosa.load(AUDIO_FILE, sr=SAMPLE_RATE)
duration = librosa.get_duration(y=waveform, sr=sr)
audio = AudioSegment.from_file(AUDIO_FILE)

# ------------------------ Load YAMNet ------------------------
print("Loading YAMNet model...")
yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = [line.strip().split(',')[2] for line in open(class_map_path).readlines()[1:]]

# ------------------------ Run Inference ------------------------
print("Running model inference...")
scores, embeddings, spectrogram = yamnet_model(waveform)
scores_np = scores.numpy()
num_frames = scores_np.shape[0]
frame_duration = len(waveform) / SAMPLE_RATE / num_frames

# ------------------------ Detect Music Segments ------------------------
music_frames = []
for i, row in enumerate(scores_np):
    top_idx = np.argmax(row)
    label = class_names[top_idx]
    score = row[top_idx]
    if "Music" in label and score > SCORE_THRESHOLD:
        start = i * frame_duration
        end = (i + 1) * frame_duration
        music_frames.append((start, end))

# ------------------------ Merge Consecutive Music Segments ------------------------
def merge_segments(segments, gap=0.1):
    if not segments:
        return []
    merged = []
    current_start, current_end = segments[0]
    for start, end in segments[1:]:
        if start - current_end <= gap:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged

merged_music = merge_segments(music_frames)

# ------------------------ Invert to Get Voice Segments ------------------------
def invert_segments(segments, total_duration):
    result = []
    prev_end = 0.0
    for start, end in segments:
        if start > prev_end:
            result.append((prev_end, start))
        prev_end = end
    if prev_end < total_duration:
        result.append((prev_end, total_duration))
    return result

all_voice_segments = invert_segments(merged_music, duration)

# ------------------------ Filter Short Voice Segments ------------------------
voice_segments = [segment for segment in all_voice_segments 
                 if segment[1] - segment[0] >= MIN_VOICE_DURATION]

skipped_segments = [segment for segment in all_voice_segments 
                   if segment[1] - segment[0] < MIN_VOICE_DURATION]

print(f"\nFound {len(all_voice_segments)} total voice segments")
print(f"Keeping {len(voice_segments)} segments longer than {MIN_VOICE_DURATION} seconds")
print(f"Skipping {len(skipped_segments)} short segments")

# ------------------------ Export Voice Segments ------------------------
summary_rows = []

def export_segment(segment, idx, start, end):
    name = f"{basename}_{idx:03d}.mp3"
    path = os.path.join(OUTPUT_DIR, name)
    segment.export(path, format="mp3")
    print(f"Saved voice segment: {path} ({round(end - start, 2)} seconds)")
    summary_rows.append({
        "type": "voice",
        "index": idx,
        "start_ms": int(start * 1000),
        "end_ms": int(end * 1000),
        "duration_sec": round(end - start, 2),
        "filename": name
    })

print("\nExporting voice segments...")
for idx, (start, end) in enumerate(voice_segments, 1):
    segment = audio[start * 1000:end * 1000]
    export_segment(segment, idx, start, end)

# ------------------------ Add Skipped Segments to Summary (but don't export) ------------------------
for idx, (start, end) in enumerate(skipped_segments, 1):
    summary_rows.append({
        "type": "skipped_voice",
        "index": idx,
        "start_ms": int(start * 1000),
        "end_ms": int(end * 1000),
        "duration_sec": round(end - start, 2),
        "filename": None
    })

# ------------------------ Write CSV Summary ------------------------
csv_path = os.path.join(OUTPUT_DIR, "segments_summary.csv")
with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["type", "index", "start_ms", "end_ms", "duration_sec", "filename"])
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\nCSV summary saved: {csv_path}")
print(f"Total voice duration: {sum(row['duration_sec'] for row in summary_rows if row['type'] == 'voice'):.2f} seconds")
print(f"Skipped voice duration: {sum(row['duration_sec'] for row in summary_rows if row['type'] == 'skipped_voice'):.2f} seconds")
