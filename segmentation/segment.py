import os
import argparse
import logging
import json
from more_itertools import chunked
from collections.abc import Iterator

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.matcher import Matcher
from whisper.utils import format_timestamp

# Define custom token extensions
Token.set_extension("can_fragment_after", default=False, force=True)
Token.set_extension("fragment_reason", default="", force=True)

def get_time_span(span: Span, timing: dict):
    if len(span) == 0:
        return None, None
        
    start_token = span[0]
    end_token = span[-1]
    try:
        while (start_token.is_punct or not timing.get(start_token.idx, None)) and start_token.i > 0:
            start_token = start_token.nbor(-1)
        while (end_token.is_punct or not timing.get(end_token.idx, None)) and end_token.i > 0:
            end_token = end_token.nbor(-1)
    except (IndexError, ValueError):
        logging.debug("Token navigation error in span: %s", span.text)
        return None, None
        
    end_index = end_token.idx
    start_index = start_token.idx
    
    if start_index not in timing or end_index not in timing:
        logging.debug("Missing timing information for span: %s", span.text)
        return None, None
        
    start, _ = timing[start_index]
    _, end = timing[end_index]
    
    return (start, end)

Span.set_extension("get_time_span", method=get_time_span, force=True)

@Language.factory("fragmenter")
class FragmenterComponent:
    def __init__(self, nlp: Language, name: str, verbal_pauses: list):
        self.nlp = nlp
        self.name = name
        self.pauses = set(verbal_pauses)

    def __call__(self, doc: Doc) -> Doc:
        return self.fragmenter(doc)

    def fragmenter(self, doc: Doc) -> Doc:
        matcher = Matcher(self.nlp.vocab)
        
        # Define patterns
        punct_pattern = [{'IS_PUNCT': True, 'ORTH': {"IN": [",", ":", ";"]}}]
        conj_pattern = [{"POS": {"IN": ["CCONJ", "SCONJ"]}}]
        clause_pattern = [{"DEP": {"IN": ["advcl", "relcl", "acl", "acl:relcl"]}}]
        
        # Add patterns to matcher
        matcher.add("punct", [punct_pattern])
        matcher.add("conj", [conj_pattern])
        matcher.add("clause", [clause_pattern])
        
        # Find matches
        matches = matcher(doc)
        
        for match_id, start, end in matches:
            rule = doc.vocab.strings[match_id]
            matched_span = doc[start:end]
            
            if rule == "punct":
                matched_span[0]._.can_fragment_after = True
                matched_span[0]._.fragment_reason = "punctuation"
            elif rule == "conj":
                if start > 0:
                    doc[start-1]._.can_fragment_after = True
                    doc[start-1]._.fragment_reason = "conjunction"
            elif rule == "clause":
                if start > 0:
                    doc[start-1]._.can_fragment_after = True
                    doc[start-1]._.fragment_reason = "clause"
        
        # Handle verbal pauses
        for token in doc:
            if token.i in self.pauses:
                token._.can_fragment_after = True
                token._.fragment_reason = "verbal pause"
        
        return doc

def load_whisper_json(file: str) -> tuple[str, dict]:
    doc_timing = {}
    doc_text = ""
    with open(file) as js:
        jsdata = json.load(js)
    for s in jsdata['segments']:
        for word_timed in s['words']:
            if word_timed['text'] == '[*]':
                continue  # Skip non-speech segments
            word = word_timed['text']
            if len(doc_text) == 0:
                word = word.lstrip()
                start_index = 0
            doc_text += word + ' '  # Add space between words
            start_index = len(doc_text) - len(word) - 1  # Account for added space
            doc_timing[start_index] = (word_timed['start'], word_timed['end'])
    return doc_text.strip(), doc_timing

def scan_for_pauses(doc_text: str, timing: dict) -> list[int]:
    pauses = []
    sorted_timings = sorted(timing.items())
    for i in range(len(sorted_timings) - 1):
        (k1, (_, end)), (k2, (start, _)) = sorted_timings[i], sorted_timings[i+1]
        gap = start - end
        if gap > 0.5:
            pauses.append(k1)
    return pauses

def divide_span(span: Span, args) -> Iterator[Span]:
    max_width = args.width
    min_width = max_width // 3
    punctuation_only = args.punctuation_only
    
    if len(span.text) <= max_width:
        yield span
        return

    # Track current chunk length
    current_chunk_start = 0
    last_good_break = None
    last_break_score = 0
    
    for i, token in enumerate(span):
        current_length = len(span[current_chunk_start:i+1].text)
        
        # Score potential break points
        if token._.can_fragment_after:
            score = 0
            is_punctuation = token.text in [",", ".", ":", ";"] or token._.fragment_reason == "punctuation"
            
            if punctuation_only and not is_punctuation:
                continue
                
            if min_width <= current_length <= max_width:
                score = 10  # Base score for acceptable length
                if token.text in [",", ".", ":", ";"]:
                    score += 5  # Prefer breaks at strong punctuation
                elif token._.fragment_reason == "punctuation":
                    score += 3  # Other punctuation
                elif not punctuation_only and token._.fragment_reason in ["conjunction", "clause"]:
                    score += 2  # Conjunctions and clauses if not punctuation-only mode
                
                # Prefer more balanced splits
                balance = 1 - abs(current_length - max_width/2) / (max_width/2)
                score += balance * 2
                
                if score > last_break_score:
                    last_good_break = i
                    last_break_score = score
        
        # Force break if chunk is too long
        if current_length > max_width and last_good_break is not None:
            yield span[current_chunk_start:last_good_break+1]
            current_chunk_start = last_good_break + 1
            last_good_break = None
            last_break_score = 0
        elif current_length > max_width * 1.5:  # Force break at current token if no good break found
            if i > current_chunk_start:
                yield span[current_chunk_start:i]
                current_chunk_start = i
            else:
                # If we can't break, include at least one token
                yield span[current_chunk_start:current_chunk_start+1]
                current_chunk_start = current_chunk_start + 1
            last_good_break = None
            last_break_score = 0
    
    # Yield remaining chunk
    if current_chunk_start < len(span):
        yield span[current_chunk_start:]

def iterate_document(doc: Doc, timing: dict, args):
    max_lines = args.lines
    last_end = 0  # Track the end time of the previous segment
    
    for sentence in doc.sents:
        for chunk in divide_span(sentence, args):
            if not chunk:  # Skip empty chunks
                continue
            subtitle = chunk.text
            if not subtitle:  # Skip empty subtitles
                continue
            
            sub_start, sub_end = chunk._.get_time_span(timing)
            if sub_start is None or sub_end is None:  # Skip chunks with missing timing
                continue
                
            # Prevent overlaps by adjusting start/end times
            if sub_start < last_end:
                sub_start = last_end + 0.001  # Add small gap
            if sub_start >= sub_end:
                sub_end = sub_start + 0.001  # Ensure minimal duration
                
            last_end = sub_end  # Update last end time
            yield sub_start, sub_end, subtitle

def write_srt(doc, timing, args):
    comma: str = ','
    for i, (start, end, text) in enumerate(iterate_document(doc, timing, args), start=1):
        ts1 = format_timestamp(start, always_include_hours=True, decimal_marker=comma)
        ts2 = format_timestamp(end, always_include_hours=True, decimal_marker=comma)
        print(f"{i}\n{ts1} --> {ts2}\n{text}\n")

def configure_spaCy(model: str, entities: str, pauses: list = []):
    nlp = spacy.load(model)
    if model.startswith('xx'):
        raise NotImplementedError("spaCy multilanguage models are not currently supported")
    nlp.add_pipe("fragmenter", config={"verbal_pauses": pauses}, last=True)
    if entities:
        ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
        ruler.from_disk(entities)
    return nlp

def main():
    parser = argparse.ArgumentParser(
                prog='subwisp',
                description='Convert a whisper .json transcript into .srt subtitles with sentences, grammatically separated where possible.')
    parser.add_argument('input_file')
    parser.add_argument('-m', '--model', help='specify spaCy model', default="en_core_web_trf")
    parser.add_argument('-e', '--entities', help='optional custom entities for spaCy (.jsonl format)', default="")
    parser.add_argument('-w', '--width', help='maximum line width', default=42, type=int)
    parser.add_argument('-l', '--lines', help='maximum lines per subtitle', default=2, type=int, choices=range(1,4))
    parser.add_argument('-d', '--debug', help='print debug information',
                        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('--verbose', help='be verbose', 
                        action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument('--punctuation-only', help='split only at punctuation', 
                        action="store_true", default=False)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    
    if not os.path.isfile(args.input_file):
        logging.error("File not found: %s", args.input_file)
        exit(-1)
    if not args.model:
        logging.error("No spacy model specified")
        exit(-1)
    if len(args.entities) > 0 and not os.path.isfile(args.entities):
        logging.error("Entities file not found: %s", args.entities)
        exit(-1)
    
    wtext, word_timing = load_whisper_json(args.input_file)
    verbal_pauses = scan_for_pauses(wtext, word_timing)
    nlp = configure_spaCy(args.model, args.entities, verbal_pauses)
    doc = nlp(wtext)
    write_srt(doc, word_timing, args)

if __name__ == '__main__':
    main()