import numpy as np
import cv2

def DrawRect(frame: np.ndarray, topLeft: tuple[int, int], size: tuple[int, int],
            color: tuple[int, int, int], thickness: int) -> None:
    # Calculate bottomRight corner based on topLeft and size (width, height)
    bottomRight = (topLeft[0] + size[1], topLeft[1] + size[0])
    cv2.rectangle(frame, topLeft, bottomRight, color, thickness)

def DrawOptFlow(img: np.ndarray, flow: np.ndarray, scale: float, color: tuple[int,int,int]) -> np.ndarray:
    h, w = img.shape[0:2]
    step = 32
    #create a grid
    x, y = np.mgrid[step//2:w:step, step//2:h:step].reshape(2, -1)
    #flow.shape is [h, w, 2] -> last dim contains f(x) and f(y)
    fx = flow[y, x, 0] * scale
    fy = flow[y, x, 1] * scale

    lines = np.vstack([x, y, x+fx, y+fy]).T #(N, x, y, x+fx, y+fy) where N is number of grid points
    lines = lines.reshape(-1, 2, 2) #each line defined as x, y and x+fx, y+fy
    lines = np.round(lines).astype(int) #round to nearest integer for pixel coordinates
    
    return cv2.polylines(img, lines, lineType=cv2.LINE_AA,
                        isClosed=False, color=color, thickness=1) #red lines
    

def MakeSubtitle(frame: np.ndarray, text: str, scaleDown: np.uint8) -> None:
    if len(text) == 0:
        return
    font = cv2.FONT_HERSHEY_COMPLEX
    frameHeight = frame.shape[0]
    margin = 50 // scaleDown
    bottomLeftPos = (10, frameHeight - margin)  # (x, y) coordinates
    fontScale = 1.2 / scaleDown
    fontColor = (255, 255, 255)  # White
    backgroundColor = (255, 0, 0)  # Blue
    thickness = 2

    textWidth, textHeight = cv2.getTextSize(text, font, fontScale, thickness)[0]

    # Calculate the top left corner of the rectangle
    topLeftCorner = (bottomLeftPos[0], bottomLeftPos[1] - textHeight - 10)
    bottomRightCorner = (bottomLeftPos[0] + textWidth + 10, bottomLeftPos[1] + 10)

    # Draw the background rectangle
    cv2.rectangle(frame, topLeftCorner, bottomRightCorner, backgroundColor, cv2.FILLED)

    # Draw the text
    cv2.putText(frame, text, bottomLeftPos, font, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)