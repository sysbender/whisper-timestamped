


Buffered Subtitle End Time Adjustment for VTT 
Ensure that each subtitle segment in the VTT file remains visible slightly longer by adding a configurable buffer (e.g., 200ms) to the end timestamp of each segment .
for example:
00:00:03.190 --> 00:00:03.610 will become 00:00:03.190 --> 00:00:03.810

command line: 
ajust.py /data/a.vtt 
will create a new file: /data/a.ajusted.vtt

in the End Time Ajust  function:
Optional configuration parameter: buffer_seconds (default = 0.2).
Add buffer_seconds to each segment’s end time in the VTT file.
Ensure no segment's end time exceeds the start time of the next segment (if one exists).
If seg[i].end + buffer > seg[i+1].start, then cap it to seg[i+1].start.