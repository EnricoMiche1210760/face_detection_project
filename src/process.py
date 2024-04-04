import os
import zipfile
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, filters, feature, color
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

def handle_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button clicked at: ({}, {})".format(x, y))

def denoise_image(image_file):
    image = img_as_float(cv2.imread(image_file))
    sigma_est = estimate_sigma(image, average_sigmas=True, channel_axis=-1)
    patch_kw = dict(patch_size=5, patch_distance=6, channel_axis=-1)
    denoised_image = denoise_nl_means(image, h=1.15 * sigma_est, sigma = sigma_est, fast_mode=True, **patch_kw)
    return denoised_image

def equalize_image(image):
    image = np.uint8(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray_image)

def difference_of_gaussian(image, show_image:bool=False):
    k = 1.6 # Gaussian blur factor
    for idx, sigma in enumerate([4, 8, 16, 32]):
        s1 = filters.gaussian(image, sigma)
        s2 = filters.gaussian(image, sigma*k)
        dog = s1 - s2
        dog = np.uint8(cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX))


        if show_image:
            cv2.imshow("DOG", dog)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return dog

def process_image(image_file, resize:bool=False, img_resize:tuple=(64, 64), diff_of_gaussian:bool=False):
    image = denoise_image(image_file)
    if resize:
        image = cv2.resize(image, img_resize)
    eq_image = equalize_image(image)
    if diff_of_gaussian:
        dog = difference_of_gaussian(eq_image)
        return dog
    return eq_image

def show_image_with_keypoints(image, keypoints):
    img = cv2.drawKeypoints(image, keypoints, None)
    cv2.imshow("Image", img)
    while True:
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            break
    cv2.destroyAllWindows()

def extract_SIFT_features(image : np.ndarray, debug:bool=False):
    sift = cv2.SIFT_create()
    kp = sift.detect(image, None)
    if debug:
        show_image_with_keypoints(image, kp)
    kp = sorted(kp, key = lambda x:x.response, reverse=True)[:SIFT_FEATURES]
    kp, des = sift.compute(image, kp)
    return (kp, des)

def extract_ORB_features(image : np.ndarray, debug:bool=False):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    if debug:
        show_image_with_keypoints(image, kp)
    #kp = sorted(kp, key = lambda x:x.response, reverse=True)[:SIFT_FEATURES]
    #kp, des = orb.compute(image, kp)
    return (kp, des)

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



def detect_faces(img_path, pipeline : object, threshold=0.5, window_size=(64, 64), step_size=(8, 8)):
    preproc_img = pipeline.named_steps['preprocess'](img_path, 'test')
    window = sliding_window(preproc_img, window_size, step_size)

    features_flattened = []
    for win in window:
        x, y, roi = win
        keypoints, features = pipeline.named_steps['extract_features'](roi)
        if features is not None:
            features_flattened = features.reshape(features.shape[0], -1)
            pca_img = pipeline.named_steps['pca']
            pca_descriptors = pca_img.transform(features_flattened)
            svm = pipeline.named_steps['svc']
            predictions = svm.predict(pca_descriptors)
            print(predictions)
            faces = []
            for kp, pred in zip(keypoints, predictions):
                if pred == 1:
                    size = kp.size
                    faces.append((int(x-size/2), int(y-size/2), int(size), int(size)))

    return faces




