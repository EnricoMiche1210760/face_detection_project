import numpy as np
import cv2
import process as pc

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
        Tuple containing the sliding window and its coordinates (x, y, window_width, window_height).
    """

    image_height, image_width = image.shape[:2]

    for y in range(0, image_height - window_size[1] + 1, step_size[1]):
        for x in range(0, image_width - window_size[0] + 1, step_size[0]):
            roi = image[y:y + window_size[1], x:x + window_size[0]] #region of interest
            yield (x, y, roi)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def extract_features(image, pipeline : object, n_keypoints, method='SIFT'):    
    
    if method == 'SIFT':
        _, features = pipeline.named_steps['extract_features'](image, n_optimal_keypoints=n_keypoints)

    elif method == 'ORB':
        _, features = pipeline.named_steps['extract_features'](image, n_keypoints=n_keypoints)

    if method == 'HOG':
        #scaler = pipeline.named_steps['normalize']
        features = pipeline.named_steps['extract_features'](image, debug=False)#, equalize=True)

    elif features is None or features.shape[0] != n_keypoints:
        return None
    
    return features

def get_image_prediction(features, pipeline : object, method='SIFT', verbose = False, threshold=0.5):    
    if(method != 'HOG'):
        features = features.flatten()
        features = np.array(features).reshape(1, -1)

    if method == 'SIFT': #or method == 'HOG':
        scaler = pipeline.named_steps['normalize']
        features = scaler.transform(features)
    svm = pipeline.named_steps['svc']
    score = svm.decision_function(features)
    score = sigmoid(svm.decision_function(features))

    y_pred = svm.predict(features)#np.where(score > threshold, 1, 0)

    if verbose:
        print(y_pred, score)

    return (y_pred, score)

def detect_faces(image, pipeline : object, method='SIFT', threshold=0.5, window_size=(96, 96), \
                 step_size=(16, 16), n_keypoints=500, resize=False, image_size=(96,96), verbose=False, notebook=False, preprocess=True):
    face_keypoints = []
    score = 0
    if window_size is None:

        if(method != 'HOG'):
            preproc_img = pipeline.named_steps['preprocess'](image, resize=resize, img_resize=image_size)
        elif preprocess:
            preproc_img = pipeline.named_steps['preprocess'](image, resize=resize, equalize=False, img_resize=image_size)
        else:
            if resize:
                image = cv2.resize(image, image_size)
            preproc_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        try:
            features = extract_features(image, pipeline, n_keypoints, method=method)
            y_pred, score = get_image_prediction(features, pipeline, method=method, threshold=threshold, verbose=verbose)
            keypoints = np.array((2, 1), dtype=int)
            if method == 'ORB':
                window_keypoints = np.array([[kp[0], kp[1]] for kp, pred in zip(keypoints, y_pred) if pred == 1])
            elif method == 'SIFT':
                window_keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp, pred in zip(keypoints, y_pred) if pred == 1])
            elif method == 'HOG':
                window_keypoints = np.array([[kp[0], kp[1]] for kp, pred in zip(keypoints, y_pred) if pred == 1])    
            if verbose and method != 'HOG':
                pc.show_image_with_keypoints(preproc_img, keypoints, notebook=notebook)
                print(window_keypoints)

            return window_keypoints, score
        except:
            return None
    
    if(method != 'HOG'):
        image = pipeline.named_steps['preprocess'](image, resize=resize, img_resize=image_size)
    elif preprocess:
        image = pipeline.named_steps['preprocess'](image, resize=resize, equalize=False, img_resize=image_size)
    else:
        if resize:
            image = cv2.resize(image, image_size)
        '''TEST'''
        image = pc.equalize_image(image)
        '''FINE TEST'''

    features = None
    for x, y, win in sliding_window(image, window_size=window_size, step_size=step_size):
        try:
            if features is None:
                features = extract_features(win, pipeline, n_keypoints, method=method)
            else:
                features = np.vstack((features, extract_features(win, pipeline, n_keypoints, method=method)))
        except Exception as e:
            print(e)
            continue
    
    y_pred, score = get_image_prediction(features, pipeline, method=method,\
                                                                threshold=threshold, verbose=verbose)

    keypoints = np.array((2, 1), dtype=int)
    if method == 'ORB':
        face_keypoints = np.array([[kp[0]+x, kp[1]+y] for kp, pred in zip(keypoints, y_pred) if pred == 1])
    elif method == 'SIFT':        
        face_keypoints = np.array([[kp.pt[0]+x, kp.pt[1]+y] for kp, pred in zip(keypoints, y_pred) if pred == 1])
    elif method == 'HOG':
        face_keypoints = np.array([[kp[0]+x, kp[1]+y] for kp, pred in zip(keypoints, y_pred) if pred == 1])    

    if verbose:
        for kp, pred in zip(face_keypoints, y_pred):
            if pred == 1:
                if method == 'ORB' or method == 'HOG':
                    kp = cv2.KeyPoint_convert(kp)
                #show_image_with_keypoints(win, keypoints, notebook=notebook)
        print(face_keypoints)

    return face_keypoints, score
