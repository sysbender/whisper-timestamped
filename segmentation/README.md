# Subtitle Segmentation Tool

This tool helps create subtitles from Whisper's word-level transcriptions by intelligently segmenting text into readable chunks while maintaining accurate timing information.

## Installation

The tool requires spaCy language models. Install the required models using:

```bash
# For English
python -m spacy download en_core_web_trf

# For French
python -m spacy download fr_core_news_sm

# For other languages, check spaCy's model list
```

## Usage

Basic command structure:
```bash
python segment.py [options] input_file.words.json
```

### Options

- `-m, --model`: Specify spaCy model (default: "en_core_web_trf")
- `-w, --width`: Maximum line width (default: 42)
- `-l, --lines`: Maximum lines per subtitle (1-3, default: 2)
- `--punctuation-only`: Split only at punctuation marks
- `-d, --debug`: Print debug information
- `--verbose`: Be more verbose
- `-f, --format`: Output format (vtt or srt, default: vtt)

### Examples

1. Basic English segmentation (outputs VTT by default):
```bash
python segment.py -m en_core_web_trf input.words.json
```

2. French segmentation with longer lines in SRT format:
```bash
python segment.py -m fr_core_news_sm -w 50 -f srt input.words.json
```

3. Split only at punctuation marks:
```bash
python segment.py -m fr_core_news_sm --punctuation-only input.words.json
```

4. Three lines per subtitle with custom width:
```bash
python segment.py -m en_core_web_trf -l 3 -w 45 input.words.json
```

### Output Formats

The tool supports two subtitle formats:

#### VTT (Default)
```vtt
WEBVTT

00:00:00.300 --> 00:00:02.040
First line of subtitle

00:00:02.420 --> 00:00:05.720
Second line of subtitle
```

#### SRT (Optional)
```srt
1
00:00:00,300 --> 00:00:02,040
First line of subtitle

2
00:00:02,420 --> 00:00:05,720
Second line of subtitle
```

### Best Practices

1. **Line Length**: Keep line width between 30-45 characters for optimal readability.
2. **Segmentation Points**:
   - By default, segments break at natural points like:
     - Punctuation marks (., ,, :, ;)
     - Clause boundaries
     - Conjunctions
   - Use `--punctuation-only` for stricter segmentation at punctuation marks only
3. **Language Models**:
   - Use transformer models (`*_trf`) for better accuracy when possible
   - Smaller models (`*_sm`) work well for faster processing
   - Match the model language to your content

### Common Issues

1. **Overlapping Timestamps**: The tool automatically fixes overlapping timestamps by adding small gaps between segments.
2. **Long Sentences**: Very long sentences are automatically split while maintaining grammatical structure.
3. **Model Selection**: If you see unexpected segmentation, try:
   - Using a different language model
   - Adjusting the width parameter
   - Using `--punctuation-only` for simpler splits

## Examples with Real Output

### Example 1: French segmentation with punctuation-only

```bash
python segment.py -m fr_core_news_sm --punctuation-only input.words.json
```

Output:
```srt
1
00:00:00,300 --> 00:00:02,040
Vous a m'orier créer un site web,

2
00:00:02,420 --> 00:00:05,720
mais vous ne savez pas vraiment comment 
vous y prendre ou par quelle boue commencer.
```

### Example 2: English with custom width

```bash
python segment.py -m en_core_web_trf -w 40 input.words.json
```

Output:
```srt
1
00:00:00,300 --> 00:00:02,040
Welcome to this tutorial about
web development

2
00:00:02,420 --> 00:00:05,720
where we'll learn the basics of HTML
and CSS.
```

## Advanced Usage

### Custom Entity Recognition

Use `-e` option to provide custom entity definitions for better segmentation:

```bash
python segment.py -m en_core_web_trf -e custom_entities.jsonl input.words.json
```

### Debug Mode

For troubleshooting segmentation issues:

```bash
python segment.py -m fr_core_news_sm -d input.words.json
```