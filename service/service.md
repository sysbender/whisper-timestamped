write a python script, will be running as a service , 

it /data folder, there are three subfolders : input, processing , output

when  there are mp3 or mp4 files  in input folder, then will be processed one by one
1. move a file to processing folder
2. if it's mp4 file, extract mp3 file from mp4 file,  
3. from mp3 file , use whisper_timestamped command to transcribe a word level timestamped json file 
4. use the segmentation/segment.py to convert json file to vtt subtitle file and put it in output folder, 
5. after the vtt file finished, clean processing folder to be ready to process next file