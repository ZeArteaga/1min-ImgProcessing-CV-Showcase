import os 
import numpy as np 
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip

def GetNextFrame(cap: cv2.VideoCapture) -> np.ndarray:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) #restart from the first frame
        ret, frame = cap.read()
    return frame

def GetImgsFromFolder(folder: str, resize: np.uint8 = 1) -> dict[str, np.ndarray]:
    imgs: dict[str, np.ndarray] = {}
    
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_COLOR)
            height, width = img.shape[:2]
            img = cv2.resize(img, (width // resize, height // resize), interpolation=cv2.INTER_LINEAR)
            imgs[filename[:-4]] = img #remove extension from filename
 
    return imgs

def addAudioToVideo(audioPath: str, videoPath: str, outputPath: str) -> None:
    video = VideoFileClip(videoPath)
    audio = AudioFileClip(audioPath)

    video = video.set_audio(audio)
    video.write_videofile(outputPath, codec='libx264', audio_codec='aac')

    video.close()
    audio.close()
