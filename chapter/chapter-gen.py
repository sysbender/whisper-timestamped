import tensorflow as tf
import numpy as np
import librosa
import os
import sys
import tensorflow_hub as hub
from pathlib import Path

# ------------------------ Command-Line Argument ------------------------
if len(sys.argv) < 3:
    print("Usage: python chapter-gen.py <audio_file> <output_dir> [min_duration]")
    sys.exit(1)

AUDIO_FILE = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

# Add command-line parameter for minimum voice segment duration (default: 2 seconds)
MIN_VOICE_DURATION = 2.0  # Default value in seconds
if len(sys.argv) >= 4:
    try:
        MIN_VOICE_DURATION = float(sys.argv[3])
        print(f"Using minimum voice segment duration: {MIN_VOICE_DURATION} seconds")
    except ValueError:
        print(f"Invalid minimum duration value. Using default: {MIN_VOICE_DURATION} seconds")
os.makedirs(OUTPUT_DIR, exist_ok=True)

basename = Path(AUDIO_FILE).stem
OUTPUT_VTT = os.path.join(OUTPUT_DIR, f"{basename}.chapters.vtt")

# ------------------------ Constants ------------------------
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
SCORE_THRESHOLD = 0.5
SAMPLE_RATE = 16000

# ------------------------ Load Audio ------------------------
print("Loading audio...")
waveform, sr = librosa.load(AUDIO_FILE, sr=SAMPLE_RATE)
duration = librosa.get_duration(y=waveform, sr=sr)

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

print(f"\nFound {len(all_voice_segments)} total voice segments")
print(f"Keeping {len(voice_segments)} segments longer than {MIN_VOICE_DURATION} seconds")
print(f"Skipping {len(all_voice_segments) - len(voice_segments)} short segments")

# ------------------------ Format Time for VTT ------------------------
def format_vtt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

# ------------------------ Write VTT File ------------------------
print(f"\nWriting VTT file: {OUTPUT_VTT}")
with open(OUTPUT_VTT, "w", encoding="utf-8") as vtt:
    vtt.write("WEBVTT\n\n")
    for idx, (start, end) in enumerate(voice_segments, 1):
        vtt.write(f"{idx}\n")  # Add cue identifier
        vtt.write(f"{format_vtt_time(start)} --> {format_vtt_time(end)}\n")
        vtt.write(f"Voice Segment {idx}\n\n")

print(f"Found {len(voice_segments)} voice segments")
print(f"VTT chapters file saved: {OUTPUT_VTT}")
print(f"Total voice duration: {sum(end - start for start, end in voice_segments):.2f} seconds")
