import os
import zipfile
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float
import numpy as np
import user_warnings as uw

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
        filename = path+list_of_images[0]
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

def process_image(image_file):
    image = denoise_image(image_file)
    eq_image = equalize_image(image)
    return eq_image

def extract_features_image(image : np.ndarray, debug=False):
    sift = cv2.SIFT_create()
    kp = sift.detect(image, None)
    if debug:
        img = cv2.drawKeypoints(image, kp, None)
        cv2.imshow("Image", img)
        while True:
            if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                break
        cv2.destroyAllWindows()


    kp = sorted(kp, key = lambda x:x.response, reverse=True)[:SIFT_FEATURES]
    kp, des = sift.compute(image, kp)
    return des



