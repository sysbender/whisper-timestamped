create subtitles from Whisper's word-level transcriptions by intelligently segmenting text into readable chunks while maintaining accurate timing information.
1. support english and french
2. support vtt and srt output format
3. split the whole subtitle by sentence, make it one sentence per line in the output
4. if a sentence contains more than 7 words, and the sentence contains comma, then split the sentence from the comma
5. don't remove or add any thing during the splitting, keep the punctuation at the end of the line, even it's used as separator for splitting


