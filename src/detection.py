import numpy as np
import cv2
import process as pc
import matplotlib.pyplot as plt
from skimage.feature import hog
import skimage.transform as transform

def sliding_window(image : np.array, window_size : tuple = (32,32), step_size : tuple = (8, 8)):
    """
    Generate sliding windows over an image.

    Parameters:
        image: numpy.ndarray
            Input image.
        window_size: tuple
            Size of the sliding window (width, height).
        step_size: tuple
            Step size for moving the window (horizontal_step, vertical_step).

    Yields:
        Tuple containing the sliding window, its coordinates and the scale factor for the pyramid level it was extracted from (x, y, roi, scale).

    """


    scales = [2, 1.5, 1.25, 1, 0.75, 0.5]
   
    for scale in scales:
        image_resized = transform.rescale(image, scale)
        image_height, image_width = image_resized.shape[:2]

        for y in range(0, image_height-window_size[1], step_size[1]):
            for x in range(0, image_width-window_size[0], step_size[0]):
                roi = image_resized[y:y + window_size[1], x:x + window_size[0]]
                if roi.shape[0] != window_size[1] or roi.shape[1] != window_size[0]:
                    continue

                yield (x, y, roi, scale)


def sigmoid(x):
    '''
    Compute the sigmoid function.

    Parameters:
        x: numpy.ndarray
            Input values.
    Returns:
        numpy.ndarray
    '''
    return 1 / (1 + np.exp(-x))

def extract_features(image, pipeline : object, n_keypoints, method='SIFT', equalize=False):
    '''
    Extract features from an image.

    Parameters:
        image: numpy.ndarray
            Input image.
        pipeline: object
            Pipeline object.
        n_keypoints: int
            Number of keypoints to extract.
        method: str
            Method to use for feature extraction, either 'SIFT', 'HOG' or 'ORB'.
        equalize: bool
            Whether to equalize the image.
    Returns:
        numpy.ndarray
    '''
    
    if method == 'SIFT':
        _, features = pipeline.named_steps['extract_features'](image, n_optimal_keypoints=n_keypoints)

    elif method == 'ORB':
        _, features = pipeline.named_steps['extract_features'](image, n_keypoints=n_keypoints)

    if method == 'HOG':
        #scaler = pipeline.named_steps['normalize']
        features = pipeline.named_steps['extract_features'](image, debug=False, equalize=equalize)

    elif features is None or features.shape[0] != n_keypoints:
        return None
    
    return features

def get_image_prediction(features, pipeline : object, method='SIFT', verbose = False, threshold=0.5):
    '''
    Get the prediction for an image.

    Parameters:
        features: numpy.ndarray
            Features extracted from the image.
        pipeline: object
            Pipeline object.
        method: str
            Method used for feature extraction ('SIFT', 'HOG' or 'ORB').
        verbose: bool
            Whether to print the prediction.
        threshold: float
            Threshold for the prediction.
    Returns:
        Tuple containing the prediction and the score.
    '''

    if(method != 'HOG'):
        features = features.flatten()
    if(features.ndim == 1):
        features = np.array(features).reshape(1, -1)

    if method == 'HOG' and features.shape[1] != 8100:
        return (0,0)
  
    if method == 'SIFT': #or method == 'HOG':
        scaler = pipeline.named_steps['normalize']
        features = scaler.transform(features)

    svm = pipeline.named_steps['svc']
    score = svm.decision_function(features)
    score = sigmoid(svm.decision_function(features))

    y_pred = np.where(score > threshold, 1, 0)

    if verbose:
        print(y_pred, score)

    return (y_pred, score)

def get_box_around_face(xs, ys, y_pred, scales, wins):
    '''
    Get the bounding box around the face.
    
    Parameters:
        xs: list
            List of x coordinates.
        ys: list
            List of y coordinates.
        y_pred: numpy.ndarray
            Predictions for each window.
        scales: list
            List of scales.
        wins: list
            List of windows.
    Returns:
        Tuple containing the bounding boxes and the keypoints.
    '''

    face_keypoints = []
    boxes = None
    for i, pred in enumerate(y_pred):
        if pred == 1:
            scale = scales[i]

            x, y = int(xs[i]//scale), int(ys[i]//scale)
            roi = wins[i]
            face_keypoints.append([roi, xs[i], ys[i]])

            if boxes is None:
                boxes = np.array([x, y, x + roi.shape[1]//scale, y + roi.shape[0]//scale])
            else:
                boxes = np.vstack((boxes, [x, y, x + roi.shape[1]//scale, y + roi.shape[0]//scale]))

    return boxes, face_keypoints


def non_max_suppression(boxes : np.ndarray, scores, overlap_threshold=.6):
    '''
    Apply non-maximum suppression to the bounding boxes.

    Parameters:
        boxes: numpy.ndarray
            Bounding boxes.
        scores: numpy.ndarray
            Scores for each bounding box.
        overlap_threshold: float
            Threshold for the overlap.
    Returns:
        Tuple containing the bounding boxes and the scores.
    '''
    if boxes is None:
        return (None, None)
    
    try:
        if(boxes.shape[1] >= 1):
            pass
    except:
        return (boxes, scores)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores) #scores

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap_boxes = np.array([xx1, yy1, xx2, yy2]).T

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
        

    return (boxes[pick].astype("int"), scores[pick])



def detect_faces(image, pipeline : object, method='SIFT', threshold=.5, window_size=(96, 96), \
                 step_size=(16, 16), n_keypoints=500, resize=False, image_size=(96,96), overlap_threshold=.6, verbose=False, notebook=False):
    '''
    Detect faces in an image.

    Parameters:
        image: numpy.ndarray
            Input image.
        pipeline: object
            Pipeline object.
        method: str
            Method to use for feature extraction ('SIFT', 'HOG' or 'ORB').
        threshold: float
            Threshold for the prediction.
        window_size: tuple
            Size of the sliding window (width, height).
        step_size: tuple
            Step size for moving the window (horizontal_step, vertical_step).
        n_keypoints: int
            Number of keypoints to extract.
        resize: bool
            Whether to resize the image.
        image_size: tuple
            Size of the image after resizing.
        overlap_threshold: float
            Threshold for the overlap.
        verbose: bool
            Whether to print the prediction.
        notebook: bool
            Whether to display the image in a notebook.
    Returns:
        Tuple containing the bounding boxes, keypoints and scores.
    '''

    boxes = []
    scores = 0

    if window_size[0] > image.shape[1] or window_size[1] > image.shape[0]:
        image = cv2.resize(image, (window_size[0], window_size[1]))

    if method != 'HOG':
        image = pipeline.named_steps['preprocess'](image, denoise=True, resize=resize, img_resize=image_size)
    else:
        image = pipeline.named_steps['preprocess'](image, denoise=True, resize=resize, img_resize=image_size)
    features = None

    xs, ys, wins, scales = zip(*sliding_window(image, window_size=window_size, step_size=step_size))

    try:
        features = np.array([extract_features(win, pipeline, n_keypoints, method=method) for win in wins])
    except Exception as e:
        print(e)
    
    y_pred, scores = get_image_prediction(features, pipeline, method=method, threshold=threshold, verbose=verbose)
    boxes, face_keypoints = get_box_around_face(xs, ys, y_pred, scales, wins)
    boxes, scores = non_max_suppression(boxes, scores[y_pred==1], overlap_threshold=overlap_threshold)


    if verbose:
        #for kp, pred in zip(face_keypoints, y_pred):
        #    if pred == 1:
        #        if method == 'ORB' or method == 'HOG':
        #            kp = cv2.KeyPoint_convert(kp)
        #        #show_image_with_keypoints(win, keypoints, notebook=notebook)
        print(scores)
        print(boxes)

    return (boxes, face_keypoints, scores)
