import numpy as np
import cv2

def Dft(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)
    return dftShift

def IdealLowPass(shape, cutoff) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2 #center row, center column. (// is floor division)
    mask = np.ones((rows, cols), dtype = np.float32)
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)
    dist = np.sqrt((x - crow)**2 + (y - ccol)**2)
    mask[dist > cutoff] = 0
    return mask

def IdealHighPass(shape, cutoff) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2 #center row, center column. (// is floor division)
    mask = np.ones((rows, cols), dtype = np.float32)
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)
    dist = np.sqrt((x - crow)**2 + (y - ccol)**2)
    mask[dist < cutoff] = 0
    return mask

def IdealBandPass(shape, cutoffLow, cutoffHigh) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2 #center row, center column. (// is floor division)
    mask = np.ones((rows, cols), dtype = np.float32)
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)
    dist = np.sqrt((x - crow)**2 + (y - ccol)**2)
    mask[dist < cutoffLow] = 0
    mask[dist > cutoffHigh] = 0
    return mask

def ApplyFilter(dftShift: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filteredDftShift = dftShift * mask[:,:,np.newaxis]
    filteredDft = np.fft.ifftshift(filteredDftShift) 
    img_back = cv2.idft(filteredDft)
    return img_back