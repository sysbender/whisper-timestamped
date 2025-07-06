#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
from datetime import datetime

def parse_timestamp(ts: str) -> float:
    """Convert VTT timestamp (HH:MM:SS.mmm) to seconds"""
    h, m, s = ts.split(':')
    # Handle milliseconds properly
    s = s.replace(',', '.')  # VTT uses comma for milliseconds
    return int(h) * 3600 + int(m) * 60 + float(s)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    # Format with exactly 3 decimal places, keep the dot for milliseconds
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def adjust_vtt_end_times(input_path: Path, output_path: Path, buffer_seconds: float = 0.2):
    """Adjust VTT subtitle end times with a buffer while ensuring no overlap."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading VTT file: {e}")
        sys.exit(1)

    output_lines = []
    timestamps = []
    
    # First pass: collect all timestamps
    for line in lines:
        if '-->' in line:
            start, end = line.strip().split(' --> ')
            timestamps.append((parse_timestamp(start), parse_timestamp(end)))
    
    # Second pass: adjust timestamps and write output
    timestamp_index = 0
    for line in lines:
        if '-->' in line:
            start, end = line.strip().split(' --> ')
            start_time = parse_timestamp(start)
            end_time = parse_timestamp(end)
            
            # Add buffer to end time
            new_end_time = end_time + buffer_seconds
            
            # Check for overlap with next subtitle
            if timestamp_index < len(timestamps) - 1:
                next_start = timestamps[timestamp_index + 1][0]
                new_end_time = min(new_end_time, next_start)
            
            # Format the new timestamp line
            new_line = f"{format_timestamp(start_time)} --> {format_timestamp(new_end_time)}\n"
            output_lines.append(new_line)
            timestamp_index += 1
        else:
            output_lines.append(line)
    
    # Write the output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        print(f"Successfully created adjusted VTT: {output_path}")
    except Exception as e:
        print(f"Error saving adjusted VTT file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Adjust VTT subtitle end times by adding a configurable buffer.'
    )
    parser.add_argument('input_vtt', 
                       help='Input VTT file path')
    parser.add_argument('--buffer', '-b', 
                       type=float, 
                       default=0.2,
                       help='Buffer time in seconds to add to end timestamps (default: 0.2)')
    parser.add_argument('--output', '-o',
                       help='Output VTT file path (default: input_name.adjusted.vtt)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_vtt)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
        
    # If output path not specified, create default name
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}.adjusted{input_path.suffix}"
    
    adjust_vtt_end_times(input_path, output_path, args.buffer)

if __name__ == "__main__":
    main()
