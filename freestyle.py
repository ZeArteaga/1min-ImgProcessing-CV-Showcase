import cv2
import mediapipe as mp
import numpy as np

class FreestyleBackgroundSub:
    def __init__(self):
        # Initialize mediapipe selfie segmentation
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)  # 0: General, 1: Landscape

    def apply_background_sub(self, frame, background_image=None, threshold=0.5) -> np.ndarray:
        if frame is not None:
            rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.segmenter.process(rgbFrame)
            mask = results.segmentation_mask
            mask = cv2.GaussianBlur(mask, (7, 7), 5)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) #expand to 3 channels
            # Debug
            #cv2.imshow('Segmentation Mask', mask)

            if background_image is not None:
                background = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
            else:
                # default background (black)
                background = np.zeros_like(frame)
            
            return np.where((mask > threshold), frame, background)
        else:
            return None


