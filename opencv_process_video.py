"""Part of skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import numpy as np
import sys
import os 

from freq_filtering import IdealLowPass, IdealHighPass, IdealBandPass, ApplyFilter, Dft
from freestyle import FreestyleBackgroundSub
from utils_draw import DrawRect, DrawOptFlow, MakeSubtitle
from utils_file_handling import GetNextFrame, GetImgsFromFolder, addAudioToVideo

def recognizeDigitString(region: np.ndarray, numTemplates: dict[str, np.ndarray], threshold: float = 0.35) -> str:
    digits = []    
    for num, numTemp in numTemplates.items():
        numResult = cv2.matchTemplate(region, numTemp, cv2.TM_CCOEFF_NORMED)
        #zero appears twice, so we need to find both
        if num == '0':
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(numResult)
            if maxVal1 > threshold:
                digits.append((num, maxLoc1[0]))
                DrawRect(region, maxLoc1, numTemp.shape[0:-1], (255, 0, 0), 1)
                # zero out the first match to find the second one
                numResult[maxLoc1[1] - numTemp.shape[0]//2 : maxLoc1[1] + numTemp.shape[0]//2,
                          maxLoc1[0] - numTemp.shape[1]//2 : maxLoc1[0] + numTemp.shape[1]//2] = 0

                # find second zero
                minVal2, maxVal2, minLoc2, maxLoc2 = cv2.minMaxLoc(numResult)
                if maxVal2 > threshold:
                    DrawRect(region, maxLoc2, numTemp.shape[0:-1], (255, 0, 0), 1)
                    digits.append((num, maxLoc2[0]))
                    
        else:
            _, numMaxVal, _, numMaxLoc = cv2.minMaxLoc(numResult)
            DrawRect(region, numMaxLoc, numTemp.shape[0:-1], (255, 0, 0), 1)
            if numMaxVal > threshold:
                digits.append((num, numMaxLoc[0]))

    # sorting by x coordinate
    digits.sort(key=lambda x: x[1])
    
    numberStr = ''.join([digit[0] for digit in digits])
    
    return numberStr

# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper
        
def main(inputVideoFile: str, outputVideoFile: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(inputVideoFile)
    fps = int(round(cap.get(5)))
    frameWidth = int(cap.get(3))
    frameHeight = int(cap.get(4))
    scaleDown = 1
    targetWidth = frameWidth // scaleDown
    targetHeigth = frameHeight // scaleDown
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter("noAudio.mp4", fourcc, fps, (targetWidth, targetHeigth))
    audio = "music.mp3"
    frame: np.ndarray = None
    prevGrey: np.ndarray = None
    opticalFlow: np.ndarray = None
    subText: str = ""

    # Set window (optional)
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', targetWidth, targetHeigth)

    #for template matching part
    mainTemplates: dict[str, np.ndarray] = GetImgsFromFolder("templates", scaleDown)
    #pop the student number label template and store in a separate variable
    snLabel = mainTemplates.pop("studentNumberLabel")
    numTemplates: dict[str, np.ndarray] = GetImgsFromFolder("templates/numbers", scaleDown)
    
    #for freestyle part
    freestyle = FreestyleBackgroundSub()
    capBackground = cv2.VideoCapture("background.mp4")
    blendFactor = 0.0
    blendStep = 0.002

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize the frame to half the target size (better performance)
            frame = cv2.resize(frame, (targetWidth, targetHeigth), interpolation=cv2.INTER_LINEAR)
            
            #Part 1
            if between(cap, 0, 2500):
                #average smoothing
                blurKernelSize: tuple[int, int] = (7, 7)
                frame = cv2.blur(frame, blurKernelSize) # 1/area * [1, 1, ...]
                subText = "Average smoothing: kernel size = " + str(blurKernelSize)

            elif between(cap, 2500, 5000):
                #sharpening
                gaussianBlurKernelSize: tuple[int, int] = (7, 7)
                gaussFrame = cv2.GaussianBlur(frame, gaussianBlurKernelSize, 3, cv2.BORDER_DEFAULT) #sigmaX=sigmaY=3
                # sharpenKernel = np.array([[0, -1, 0], 
                #                          [-1, 5, -1], 
                #                           [0, -1, 0]])
                # sharpframe = cv2.filter2D(frame, -1, sharpenKernel)
                frame = cv2.addWeighted(frame, 1.5, gaussFrame, -0.5, 0)
                
                subText = "Sharpening"

            elif between(cap, 5000, 7500):
                #Sobel
                thresh = 190
                gaussFrame = cv2.GaussianBlur(frame, (3, 3), 1, cv2.BORDER_DEFAULT)
                greyFrame = cv2.cvtColor(gaussFrame, cv2.COLOR_BGR2GRAY)
                frameDx = cv2.Sobel(greyFrame, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
                frameDy = cv2.Sobel(greyFrame, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)
                edges = cv2.magnitude(frameDx, frameDy)  # Calculate gradient magnitude
                _, frame = cv2.threshold(edges, thresh, 255, type=cv2.THRESH_TOZERO)

                subText = "Edge detection - Sobel (threshold = " + str(thresh) + ")"

            elif between(cap, 7500, 10000):
                #canny
                lowThresh = 180
                highThresh = 230
                gaussFrame = cv2.GaussianBlur(frame, (3, 3), 1, cv2.BORDER_DEFAULT)
                greyFrame = cv2.cvtColor(gaussFrame, cv2.COLOR_BGR2GRAY)
                frame = cv2.Canny(greyFrame, lowThresh, highThresh, apertureSize=3, L2gradient=True) #default sobel kernel size 3
                subText = "Edge detection - Canny (threshold = " + str(lowThresh) + ", " + str(highThresh) + ")"
            
            #Part 2
            elif between(cap, 10000, 12500):
                dftShift = Dft(frame)
                real = dftShift[:, :, 0]
                imag = dftShift[:, :, 1]
                magnitudeLog = np.log(1 + cv2.magnitude(real, imag))

                frame = magnitudeLog
                subText = "DFT log-magnitude"
            
            elif between(cap, 12500, 15000):
                cutoff = 600//scaleDown
                dftShift = Dft(frame)
                mask = IdealLowPass(dftShift.shape[0:2], cutoff)
                filtFrame = ApplyFilter(dftShift, mask)
                #calculate mag from real and imaginary parts
                frame = cv2.magnitude(filtFrame[:, :, 0], filtFrame[:, :, 1])  
                subText = "Ideal G/W Low-pass filter (cutoff = " + str(cutoff) + " pixels)"

            elif between(cap, 15000, 17500):
                cutoff = 1000//scaleDown
                dftShift = Dft(frame)
                mask = IdealHighPass(dftShift.shape[0:2], cutoff)
                filtFrame = ApplyFilter(dftShift, mask)
                frame = cv2.magnitude(filtFrame[:, :, 0], filtFrame[:, :, 1])
                subText = "Ideal G/W High-pass filter (cutoff = " + str(cutoff) + " pixels)"

            elif between(cap, 17500, 20000):
                cutoffLow = 600//scaleDown
                cutoffHigh = 800//scaleDown
                dftShift = Dft(frame)
                mask = IdealBandPass(dftShift.shape[0:2], cutoffLow, cutoffHigh)
                filtFrame = ApplyFilter(dftShift, mask)
                frame = cv2.magnitude(filtFrame[:, :, 0], filtFrame[:, :, 1])
                subText = "Ideal G/W Band-pass filter (cutoff = " + str(cutoffLow) + " to " + str(cutoffHigh) + " pixels)"

            #Part 3
            elif between(cap, 20000, 35000):

                for name, template in mainTemplates.items():
                    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
                    _, maxVal, _, maxLoc = cv2.minMaxLoc(result)

                    if maxVal > 0.35: #discard really low matches
                        if name == "studentNumber":
                            #template holds both the label and the number, maxLoc is the top left pixel
                            cv2.matchTemplate(frame, snLabel, cv2.TM_CCOEFF_NORMED)                            
                            _, _, _, maxLocLabel = cv2.minMaxLoc(result)
                            DrawRect(frame, maxLocLabel, snLabel.shape[0:-1], (0, 0, 255), 1) #draw rectangle around label
                            
                            #actual number starts further right, so we shift by the width of the label
                            snLabelWidth = snLabel.shape[1]
                            snNumberWidth = template.shape[1] - snLabelWidth
                            snTopLeft = (maxLoc[0] + snLabelWidth, maxLoc[1])
                            snBottomRight = (snTopLeft[0] + snNumberWidth, maxLoc[1] + template.shape[0])
                            ocrRegion = frame[snTopLeft[1]:snBottomRight[1], snTopLeft[0]:snBottomRight[0]]
                            #OCR
                            DrawRect(frame, snTopLeft, ocrRegion.shape[0:-1], (0, 255, 0), 1) #draw rectangle around ocr region
                            numberStr = recognizeDigitString(ocrRegion, numTemplates)
                        else:
                            DrawRect(frame, maxLoc, template.shape[0:-1], (0,0,255), 2) #draw rectangle for other elements
                
                subText = "Template matching - student number: " + numberStr

            elif between(cap, 35000, 40000):
                greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if(prevGrey is None):
                    prevGrey = greyFrame
                
                opticalFlow = cv2.calcOpticalFlowFarneback(prevGrey, greyFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0) #default values
                frame = DrawOptFlow(frame, opticalFlow, 1.5, (0, 0, 255))
                prevGrey = greyFrame #for next iteration

                subText = "Optical Flow"

            elif between(cap, 40000, 40750):
                pass
            elif between(cap, 40750, 59000):
                background = GetNextFrame(capBackground)
                frame = freestyle.apply_background_sub(frame, background, 0.4)
                
                #slowly transition to black and white
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grey = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
                frame = cv2.addWeighted(frame, 1 - blendFactor, grey, blendFactor, 0)
                # Increase the blending factor
                blendFactor = min(blendFactor + blendStep, 1.0)
                subText = "Happy Halloween: using MediaPipe Selfie Segmentation"
            else:
                frame = np.zeros_like(frame) #cuts to black

            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = cv2.convertScaleAbs(frame) # clip to 0-255 range and convert to uint8
            if len(frame.shape) == 2: #if grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR format for output
            
            MakeSubtitle(frame, subText, scaleDown)
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Video', frame)
    
            cv2.waitKey(10)
            #we can interrupt with CTRL+C
        else:
            break

     # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    
    addAudioToVideo(audio, "noAudio.mp4", outputVideoFile)
    # delete temp video file
    os.remove("noAudio.mp4")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)
