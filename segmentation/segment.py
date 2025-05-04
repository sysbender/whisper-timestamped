import json
import argparse
import spacy
import spacy.cli
from whisper.utils import format_timestamp
import sys
from pathlib import Path

def detect_language(text: str) -> str:
    """Detect if text is primarily English or French using spacy language models"""
    def ensure_model(model_name: str):
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model '{model_name}'...", file=sys.stderr)
            spacy.cli.download(model_name)
            return spacy.load(model_name)

    # Load both models using the helper function
    nlp_en = ensure_model('en_core_web_sm')
    nlp_fr = ensure_model('fr_core_news_sm')
        
    # Process a sample of the text (first 1000 characters) for efficiency
    sample_text = text[:1000]
    
    # Get word counts for each language (excluding stop words and punctuation)
    en_doc = nlp_en(sample_text)
    fr_doc = nlp_fr(sample_text)
    
    # Count words that are recognized as proper words in each language
    en_words = len([token for token in en_doc if token.is_alpha and token.has_vector])
    fr_words = len([token for token in fr_doc if token.is_alpha and token.has_vector])
    
    # Add extra weight to words with accents for French detection
    fr_accent_bonus = len([token for token in fr_doc if token.is_alpha and any(c in 'éèêëàâäôöûüçîï' for c in token.text.lower())])
    fr_words += fr_accent_bonus
    
    return 'fr' if fr_words > en_words else 'en'

def load_whisper_json(file: str) -> tuple[str, dict]:
    doc_timing = {}
    doc_text = ""
    with open(file) as js:
        jsdata = json.load(js)
    for s in jsdata['segments']:
        for word_timed in s['words']:
            if word_timed['text'] == '[*]':
                continue
            word = word_timed['text']
            if len(doc_text) == 0:
                word = word.lstrip()
                start_index = 0
            doc_text += word + ' '
            start_index = len(doc_text) - len(word) - 1
            doc_timing[start_index] = (word_timed['start'], word_timed['end'])
    return doc_text.strip(), doc_timing

def get_segment_timing(text: str, start_idx: int, end_idx: int, timing: dict) -> tuple[float, float]:
    # Find the first and last timed indices within the segment
    start_time = None
    end_time = None
    
    for idx in range(start_idx, end_idx + 1):
        if idx in timing:
            if start_time is None:
                start_time = timing[idx][0]
            end_time = timing[idx][1]
    
    return start_time, end_time

def process_text(text: str, timing: dict, nlp) -> list[tuple[float, float, str]]:
    doc = nlp(text)
    segments = []
    
    for sent in doc.sents:
        text = sent.text.strip()
        words = text.split()
        
        # If sentence has more than 7 words and contains comma, split at comma
        if len(words) > 7 and ',' in text:
            comma_parts = text.split(',')
            start_idx = sent.start_char
            
            for i, part in enumerate(comma_parts):
                part = part.strip()
                if not part:
                    continue
                    
                end_idx = start_idx + len(part)
                if i < len(comma_parts) - 1:
                    part += ','  # Keep the comma as specified
                
                start_time, end_time = get_segment_timing(text, start_idx, end_idx, timing)
                if start_time is not None and end_time is not None:
                    segments.append((start_time, end_time, part))
                
                start_idx = end_idx + 1  # +1 to skip the comma
        else:
            start_time, end_time = get_segment_timing(text, sent.start_char, sent.end_char, timing)
            if start_time is not None and end_time is not None:
                segments.append((start_time, end_time, text))
    
    return segments

def write_srt(segments, file=sys.stdout):
    for i, (start, end, text) in enumerate(segments, 1):
        start_str = format_timestamp(start, always_include_hours=True, decimal_marker=',')
        end_str = format_timestamp(end, always_include_hours=True, decimal_marker=',')
        print(f"{i}\n{start_str} --> {end_str}\n{text}\n", file=file)

def write_vtt(segments, file=sys.stdout):
    print("WEBVTT\n", file=file)
    for start, end, text in segments:
        start_str = format_timestamp(start, always_include_hours=True, decimal_marker='.')
        end_str = format_timestamp(end, always_include_hours=True, decimal_marker='.')
        print(f"{start_str} --> {end_str}\n{text}\n", file=file)

def main():
    parser = argparse.ArgumentParser(description='Convert Whisper JSON to subtitles with simple segmentation')
    parser.add_argument('input_file', help='Input JSON file from Whisper')
    parser.add_argument('--lang', choices=['en', 'fr'], help='Language (en or fr). If not specified, will auto-detect')
    parser.add_argument('--format', choices=['vtt', 'srt'], default='vtt', help='Output format')
    parser.add_argument('-o', '--output', help='Output file path. If not specified, writes to stdout')
    
    args = parser.parse_args()
    
    try:
        # Process the file to get text for language detection if needed
        text, timing = load_whisper_json(args.input_file)
        
        # Detect or use specified language
        lang = args.lang if args.lang else detect_language(text)
        print(f"Using language: {lang}", file=sys.stderr)
        
        # Load appropriate spaCy model
        model_name = 'en_core_web_sm' if lang == 'en' else 'fr_core_news_sm'
        def ensure_model(model_name: str):
            try:
                return spacy.load(model_name)
            except OSError:
                print(f"Downloading spaCy model '{model_name}'...", file=sys.stderr)
                spacy.cli.download(model_name)
                return spacy.load(model_name)
        
        nlp = ensure_model(model_name)
        
        # Process segments
        segments = process_text(text, timing, nlp)
        
        # Handle output file
        output_file = None
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = open(output_path, 'w', encoding='utf-8')
        
        try:
            # Output in requested format
            if args.format == 'srt':
                write_srt(segments, output_file if output_file else sys.stdout)
            else:
                write_vtt(segments, output_file if output_file else sys.stdout)
        finally:
            if output_file:
                output_file.close()
                
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()