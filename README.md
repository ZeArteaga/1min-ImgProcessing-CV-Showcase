# Image-Processing-CV-ShowOff

This repository contains my submission for an individual assignment in Image Processing and Computer Vision, completed during my exchange at the University of Twente. The project applies various Img Processing and CV techniques to create a one-minute video, organized into distinct sections that demonstrate learned technical skills.

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
  
## Project Structure
The final video is divided into four main sections:

1. **Spatial Domain Effects (0-10s)**: Applies spatial image processing techniques to create special effects.
2. **Frequency Domain Effects (10-20s)**: Demonstrates effects using Fourier Transform methods in the frequency domain.
3. **Object Detection and Tracking (20-40s)**: Detects and tracks specific objects in the frame, including my student ID elements. In this section we weren't allowed to use actual OCR engines so it is not really accurate in detecting strings of digits.
4. **Freestyle Segment (40-60s)**: A surprise creative section showcasing additional CV techniques.

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- MoviePy

## Usage
**for everything stated below try ```python3``` if ```python```does not work.**

1. (**Recommended but Optional**) Creating a virtual environment helps isolate project dependencies. To create one:
   ```bash
   # Navigate to your project directory
   cd path/to/your/project

   # Create the virtual environment
   python -m venv .venv

   # Activate the virtual environment
   # On macOS/Linux
   source .venv/bin/activate

   # On Windows
   .venv\Scripts\activate

2. Install requirements:
```
pip install -r requirements.txt
```

3. Set scaleDown variable for real time resizing and faster processing (default 1). Then run the script (include file extensions):
    ```shell script
   $ python opencv_process_video.py -i <PATH_TO_INPUT_VIDEO_FILE> -o <PATH_TO_OUTPUT_VIDEO_FILE>
    ``` 
4. wait for the script to finish processing the video

5. (**Optional**) Instead of scaling down, use ffmpeg CLI tool instead to downsample the video. Example:
    ```shell script
   $ ffmpeg -i output.mp4 -c:v libx265 -b:v 4000k -preset slow -crf 28 -c:a aac -b:a 128k output_downsampled.mp4
    ```
