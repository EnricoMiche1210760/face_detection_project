import os
import zipfile
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float
from skimage.feature import ORB, hog
import numpy as np
import user_warnings as uw
import matplotlib.pyplot as plt

zip_file = "img_align_celeba.zip"
DATA_PATH = "../data"
SIFT_FEATURES = 128

def extract_dataset(path: str = None, folder:str = "single_folder"):
    if not path:
        path = DATA_PATH
    if not os.path.exists(path):
        print("Extracting dataset...")
        with zipfile.ZipFile(DATA_PATH+'/'+zip_file, 'r') as zip_ref:
            zip_ref.extractall(DATA_PATH)
    else:
        if folder == "single_folder":
            print("Dataset already extracted")
            return None
        elif folder == "multi_folders" or folder == "multiple":
            folder_list = os.listdir(path)
            return folder_list
        else:
            raise ValueError("Invalid folder type")


def load_images(path:str, number_of_images=100, random_seed=42):
    if not os.path.exists(path):
        extract_dataset()
    list_of_images = os.listdir(path)
    if number_of_images > len(list_of_images):
        uw.fxn()
        return list_of_images
    
    np.random.seed(random_seed)
    images = np.random.choice(list_of_images, number_of_images)
    return images

def load_image(path:str, filename=None):
    if filename is None:
        if not os.path.exists(path):
            extract_dataset()
        list_of_images = os.listdir(path)
        filename = path+'/'+list_of_images[0]
        print("Loaded image: ", filename)
    else:
        if filename not in os.listdir(path):
            print("File not found")
            return None
    return filename

def extract_patches(img_list, size:tuple=(96,96), n_patches=5000, random_seed=7):
    np.random.seed(random_seed)
    patches = []
    n_patches = round(n_patches / len(img_list))-1 
    print(n_patches)
    for img in img_list:
        image = cv2.imread(img)
        if image.shape[0] != 0 and image.shape[1] != 0:
            patches.append(image)
        for i in range(n_patches):
            width= image.shape[0]-size[0]
            heigth = image.shape[1]-size[1]
            if width < 0 or heigth < 0:
                continue
            if width > 0: 
                x = np.random.randint(0, width)
            else:
                x = 0
            if heigth > 0:
                y = np.random.randint(0, heigth)
            else:
                y = 0
            patch = image[y:y+size[1], x:x+size[0]]
            if patch.shape[0] == 0 or patch.shape[1] == 0:
                continue
            patches.append(patch)
    return patches


def handle_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button clicked at: ({}, {})".format(x, y))

def denoise_image(image):
    image = img_as_float(image)
    denoised_image = cv2.GaussianBlur(image, (15, 15), 0)
    return denoised_image

def equalize_image(image):
    image = np.uint8(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray_image)

def process_image(image, resize:bool=False, img_resize:tuple=(64, 64), diff_of_gaussian:bool=False):
    img = denoise_image(image)
    if resize:
        img = cv2.resize(img, img_resize)
    eq_image = equalize_image(img)
    if diff_of_gaussian:
        thresh = cv2.adaptiveThreshold(eq_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh
    return eq_image

def show_image_with_keypoints(image, keypoints, notebook=False):
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, image)

    if notebook:
        plt.imshow(img_with_keypoints)
        plt.show()
        
    else:
        cv2.imshow("Image", img_with_keypoints)
        while True:
            if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                break
        cv2.destroyAllWindows()

def extract_SIFT_features(image : np.ndarray, n_optimal_keypoints=128, debug:bool=False):
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    kpt, des = sift.detectAndCompute(image, None)
    if debug:
        print(kpt)
        show_image_with_keypoints(image, kpt)
    
    if len(kpt) < n_optimal_keypoints:
        return (kpt, des)

    sorted_indices = np.argsort([kp.response for kp in kpt])[::-1]
    kpt = [kpt[i] for i in sorted_indices[:n_optimal_keypoints]]
    des = des[sorted_indices[:n_optimal_keypoints]]

    return (kpt, des)

def extract_ORB_features(image : np.ndarray, n_keypoints:int=500, debug:bool=False):
    orb = ORB(n_keypoints=n_keypoints)
    try:
        orb.detect_and_extract(image)
    except:
        return (None, None)
    kp = orb.keypoints
    des = orb.descriptors
    kpt = cv2.KeyPoint_convert(kp)
    if debug:
        show_image_with_keypoints(image, kpt)
    return (kp, des)

def extract_HOG_features(image, cell_size=(8, 8), block_size=(3, 3), nbins=9):
    #image = cv2.equalizeHist(image)
    features = hog(image, orientations=nbins, pixels_per_cell=cell_size,
                              cells_per_block=block_size, block_norm='L2-Hys')
   
    return features


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

def get_image_prediction(image, pipeline : object, n_keypoints, method='SIFT', verbose = False, threshold=0.5):
    if method == 'SIFT':
        keypoints, features = pipeline.named_steps['extract_features'](image, n_optimal_keypoints=n_keypoints)
        scaler = pipeline.named_steps['normalize']

    elif method == 'ORB':
        keypoints, features = pipeline.named_steps['extract_features'](image, n_keypoints=n_keypoints)

    if method == 'HOG':
        scaler = pipeline.named_steps['normalize']
        features = pipeline.named_steps['extract_features'](image)
        keypoints = np.zeros((1, 2), dtype=np.float32)

    elif features is None or features.shape[0] != n_keypoints:
        return None
    
    if(method != 'HOG'):
        features = features.flatten()
    features = np.array(features).reshape(1, -1)

    if method == 'SIFT' or method == 'HOG':
        features = scaler.transform(features)
   
    svm = pipeline.named_steps['svc']
    score = svm.decision_function(features)
    score = sigmoid(svm.decision_function(features))

    y_pred = np.where(score > threshold, 1, 0)

    if verbose:
        print(y_pred, score)

    return (y_pred, score, keypoints)

def detect_faces(image, pipeline : object, method='SIFT', threshold=0.5, window_size=(96, 96), \
                 step_size=(16, 16), n_keypoints=500, resize=False, image_size=(96,96), verbose=False, notebook=False):
    face_keypoints = []
    score = 0
    if window_size is None:

        preproc_img = pipeline.named_steps['preprocess'](image, resize=resize, img_resize=image_size)

        try:
            y_pred, score, keypoints = get_image_prediction(preproc_img, pipeline, n_keypoints,\
                                                                       method=method, threshold=threshold, verbose=verbose)
            if method == 'ORB':
                window_keypoints = np.array([[kp[0], kp[1]] for kp, pred in zip(keypoints, y_pred) if pred == 1])
            elif method == 'SIFT':
                window_keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp, pred in zip(keypoints, y_pred) if pred == 1])
            elif method == 'HOG':
                window_keypoints = np.array([[kp[0], kp[1]] for kp, pred in zip(keypoints, y_pred) if pred == 1])    
            if verbose and method != 'HOG':
                show_image_with_keypoints(preproc_img, keypoints, notebook=notebook)
                print(window_keypoints)

            return window_keypoints, score
        except:
            return None
    
    image = pipeline.named_steps['preprocess'](image, resize=resize, img_resize=image_size)

    for x, y, win in sliding_window(image, window_size=window_size, step_size=step_size):
        try:
            y_pred, score, keypoints = get_image_prediction(win, pipeline, n_keypoints, method=method,\
                                                                       threshold=threshold, verbose=verbose)
        except:
            continue
        
        if method == 'ORB':
            window_keypoints = np.array([[kp[0]+x, kp[1]+y] for kp, pred in zip(keypoints, y_pred) if pred == 1])
        elif method == 'SIFT':        
            window_keypoints = np.array([[kp.pt[0]+x, kp.pt[1]+y] for kp, pred in zip(keypoints, y_pred) if pred == 1])
        elif method == 'HOG':
            window_keypoints = np.array([[kp[0]+x, kp[1]+y] for kp, pred in zip(keypoints, y_pred) if pred == 1])    

        if verbose:
            if method == 'ORB' or method == 'HOG':
                keypoints = cv2.KeyPoint_convert(keypoints)
            show_image_with_keypoints(win, keypoints, notebook=notebook)
            print(window_keypoints)


        face_keypoints.extend(window_keypoints)

    return face_keypoints, score




