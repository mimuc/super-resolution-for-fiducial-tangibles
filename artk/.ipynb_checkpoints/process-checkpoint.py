import numpy as np
import cv2

def getNorm (e):
    if (e is None):
        print("utils.py getNorm Error #001")
        return None
    e = e.copy()
    e[e<0.0] = 0.0
    normalizedImg = e.copy()#np.empty(e.shape, np.float32)
    
    cv2.normalize(e,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    return normalizedImg


def detect_blobs(image, min_cutoff = 20, blob_threshold = 220, verbose=False):
    if (image is None):
        print("detect_blobs() Error #001")
        return None

    lstBlob  = []
    lstMin = []
    lstMax = []
    lstMean = []
    lstXY = []
    
    image = image.clip(0,255)    
    if np.any(image > min_cutoff):
        image = getNorm(image).clip(0,255).astype(np.uint8)
        large = np.pad(image, 1)
        temp, thresh = cv2.threshold(cv2.bitwise_not(large), blob_threshold, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [a for a in contours if cv2.contourArea(a) > 2 and cv2.contourArea(a) < ((large.shape[0]-1) * (large.shape[1]-1))]
        
        contours.sort(key = lambda a: cv2.contourArea(a))
        for max_contour in contours:
            xmax, ymax = np.max(max_contour.reshape(len(max_contour),2), axis=0)
            xmin, ymin = np.min(max_contour.reshape(len(max_contour),2), axis=0)
            xmax, ymax = min(xmax + 1, large.shape[1]), min(ymax + 1, large.shape[0])
            xmin, ymin = max(xmin, 0), max(ymin, 0)
            blob = large[ymin:ymax,xmin:xmax].astype(np.ubyte)
            
            M = cv2.moments(max_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])          
            
            lstBlob.append(blob)
            lstMin.append(xmax-xmin)
            lstMax.append(ymax-ymin)
            lstXY.append([cX, cY])
        return lstBlob, lstMin, lstMax, lstXY
    else:
        if verbose:
            print("detect_blobs() WARNING: image values are smaller than cutoff. cutoff value: %.01f" % cutoff) 
        return [], [], [], []

    
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    if pad_width[1] != 0:
        vector[-pad_width[1]:] = pad_value
        

def getBlobBasedOnXYSizeOld(matrix, xy, size):
    #print("getBlobBasedOnXYSize() XY", xy)
    xy = np.round(xy).astype(int)
    size = np.ceil(size / 2).astype(int)
    xMax = xy[0] + size[1] + 1 
    yMax = xy[1] + size[0] + 1 
    xMin = xy[0] - size[1] - 1
    yMin = xy[1] - size[0] - 1
    
    lowerY, upperY = max(yMin, 0), min(yMax, matrix.shape[0])
    lowerX, upperX = max(xMin, 0), min(xMax, matrix.shape[1])
    
    #size = (size * 2.0).astype(int)
    #print("getBlobBasedOnXYSize() Area", xy, 'Size',size ,lowerY,upperY, lowerX,upperX)
    blob = matrix[lowerY:upperY, lowerX:upperX]
    
    #print(blob.shape)
    
    if blob.shape == ((size[0] + 1)*2, (size[1] + 1)*2):
        return blob
    
    if  yMin < lowerY:
        blob = np.pad(blob, ((np.abs(yMin), 0), (0, 0)), pad_with, padder=0)
    if  yMax > upperY:
        blob = np.pad(blob, ((0, yMax - upperY), (0, 0)), pad_with, padder=0)

    if  xMin < lowerX:
        blob = np.pad(blob, ((0, 0), (np.abs(xMin), 0)), pad_with, padder=0)
    if  xMax > upperX:
        blob = np.pad(blob, ((0, 0), (0, xMax - upperX)), pad_with, padder=0)
    
    return blob

def getBlobBasedOnXYSize(matrix, xy, size):
    #print("getBlobBasedOnXYSize() XY", xy)
    xy = np.round(xy).astype(int)
    size = np.ceil(size / 2).astype(int)
    xMax = xy[0] + size[1] + 1 
    yMax = xy[1] + size[0] + 1 
    xMin = xy[0] - size[1] - 1
    yMin = xy[1] - size[0] - 1
    
    
    lowerY, upperY = max(yMin, 0), min(yMax, matrix.shape[0])
    lowerX, upperX = max(xMin, 0), min(xMax, matrix.shape[1])
     
    #size = (size * 2.0).astype(int)
    #print("getBlobBasedOnXYSize() Area", xy, 'Size',size ,lowerY,upperY, lowerX,upperX)
    blob = matrix[lowerY:upperY, lowerX:upperX]
    
    template = np.zeros([(size[0]+1)*2,(size[1]+1)*2])
    
    try:
        template[0:(size[0]+1)*2, 0:(size[1]+1)*2] = blob
    except ValueError:
        y_lower_boundary = np.abs(min(yMin,0))
        y_upper_boundary = (size[1]+1)*2 + min(0,matrix.shape[0]-yMax)
        x_lower_boundary = np.abs(min(xMin,0))
        x_upper_boundary = (size[0]+1)*2 + min(0,matrix.shape[1]-xMax)


        template[y_lower_boundary:y_upper_boundary, x_lower_boundary:x_upper_boundary] = blob

    return template.astype(np.float32)
    
"""
 number_of_iterations: Specify the number of iterations.
 terminationEps: Specify the threshold of the increment in the correlation coefficient between two iterations
 gaussFilterSize : An optional value indicating size of gaussian blur filter. Has to be greater than 2
"""
def aligneImageECC(im1, im2, numberOfIterations = 100, terminationEps = 1e-10, gaussFilterSize = 3, warp_mode = cv2.MOTION_EUCLIDEAN):
    # Find size of image1
    sz = im1.shape

    
    # Define the motion model
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, numberOfIterations,  terminationEps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1,im2, warp_matrix, warp_mode, criteria, None, gaussFilterSize)

    # Use warpAffine for Translation, Euclidean and Affine
    return cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    
def aligneCentre(AprilTagPixelSizeInMM, PixelSizeInMM, markerPixels, img):
    shape = img.shape
    size = int(np.round(AprilTagPixelSizeInMM / PixelSizeInMM * markerPixels))
    clear = np.zeros(shape, np.float32)
    xy = np.round(np.array(shape)/2).astype(int)
    dim = np.array([xy - size/2, xy + size/2]).astype(int)
    clear[dim[0,0]: dim[1,0], dim[0,1]: dim[1,1]] = np.ones((size, size), np.float32) * img.mean()
    #clear = clear.astype(np.float32)
    imgA = aligneImageECC(clear, img)
    return imgA

def prepareRealMarker(marker, params):
    marker = np.copy(marker)
    marker[marker>params["CutOffReal"]] = params["CutOffReal"]
    return getNorm(marker) / 255.0


def prepareForML (sim, outputSize=(22,22)):
    _, _, _, lstXY = detect_blobs(sim * 255)
    sim = getBlobBasedOnXYSize(sim, lstXY[-1], outputSize)
    
    #sim = process.alignCentre(AprilTagPixelSizeInMM, PixelSizeInMM, markerPixels, sim)
    return getNorm(sim) / 255.0
