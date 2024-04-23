import cv2
from skimage import img_as_float, transform
from skimage.feature import ORB, hog
import numpy as np
import matplotlib.pyplot as plt

SIFT_FEATURES = 128

def extract_patches(img_list, size:tuple=(96,96), n_patches=5000, random_seed=7):
    '''
    Extract patches from a list of images.

    Parameters:
        img_list: list
            List of image paths.
        size: tuple
            Size of the patches (width, height).
        n_patches: int
            Number of patches to extract.
        random_seed: int
            Random seed for reproducibility.
    Returns:
        patches: numpy.ndarray

    '''
    np.random.seed(random_seed)
    patches = np.empty((n_patches, size[0], size[1], 3), dtype=np.uint8)
    patch_for_image = round(n_patches / len(img_list))-1 

    tmp_len = 0

    for img in img_list:
        image = cv2.imread(img)
        if image.shape[0] != 0 and image.shape[1] != 0:
            image = cv2.resize(image, (size[0], size[1]))
            patches[tmp_len, ...] = image
            tmp_len += 1
        if(tmp_len >= n_patches):
            return patches

    for img in img_list:
        image = cv2.imread(img)
        for _ in range(patch_for_image):
            height = image.shape[0]-size[0]
            width = image.shape[1]-size[1]
            if width < 0 or height < 0:
                continue
            if width > 0: 
                x = np.random.randint(0, width)
            else:
                x = 0
            if height > 0:
                y = np.random.randint(0, height)
            else:
                y = 0
            patch = image[y:y+size[0], x:x+size[1]]
            if patch.shape[0] < size[0] or patch.shape[1] < size[1]:
                continue
            patches[tmp_len, ...] = patch
            tmp_len += 1
        if(tmp_len >= n_patches):
            break

    if tmp_len < n_patches:
        n_missing = n_patches - tmp_len
        np.random.seed(random_seed)
        for _ in range(0, n_missing):
            random_index = np.random.randint(0, len(img_list))
            img_tmp = cv2.rotate(patches[random_index], cv2.ROTATE_90_CLOCKWISE)
            patches[tmp_len, ...] = img_tmp
            tmp_len += 1
    return patches

def equalize_image(image, normalize=False):
    '''
    Equalize an image.
    
    Parameters:
        image: numpy.ndarray
            Input image.
        normalize: bool
            Whether to normalize the image.
    Returns:
        equalized image: numpy.ndarray
    '''
    if normalize:
        image = np.uint8(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX))
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.equalizeHist(image)
    return image

def denoise_image(image):
    '''
    Denoise an image using Gaussian blur.

    Parameters:
        image: numpy.ndarray
            Input image.
    Returns:
        denoised image: numpy.ndarray
    '''
    image = img_as_float(image)
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised_image

def process_image(image, denoise:bool=False, equalize:bool=True, resize:bool=False, img_resize:tuple=(64, 64), 
                  diff_of_gaussian:bool=False):
    '''
    Process an image, converting it to grayscale, resizing it, denoising it, equalizing it and applying a difference of Gaussian filter.

    Parameters:
        image: numpy.ndarray
            Input image.
        denoise: bool
            Whether to denoise the image.
        equalize: bool
            Whether to equalize the image.
        resize: bool
            Whether to resize the image.
        img_resize: tuple
            Size of the resized image.
        diff_of_gaussian: bool
            Whether to apply a difference of Gaussian filter.
    Returns:
        processed image: numpy.ndarray
    '''

    if denoise:
        image = denoise_image(image)
        image = np.uint8(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX))
    
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if resize:
        image = cv2.resize(image, img_resize)

    if equalize:
        image = equalize_image(image)

    if diff_of_gaussian:
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return image

def show_image_with_keypoints(image, keypoints, notebook=False):
    '''
    Show an image with keypoints.

    Parameters:
        image: numpy.ndarray
            Input image.
        keypoints: list
            List of keypoints.
        notebook: bool
            Whether to display the image in a notebook or in a window.
    '''

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
    '''
    Extract SIFT features from an image.

    Parameters:
        image: numpy.ndarray
            Input image.
        n_optimal_keypoints: int
            Number of optimal keypoints to extract.
        debug: bool
            Whether to show the image with keypoints.
    Returns:
        keypoints: list
        descriptors: numpy.ndarray
    '''

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
    '''
    Extract ORB features from an image.

    Parameters:
        image: numpy.ndarray
            Input image.
        n_keypoints: int
            Number of keypoints to extract.
        debug: bool
            Whether to show the image with keypoints.
    Returns:
        keypoints: list
        descriptors: numpy.ndarray
    '''
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

def extract_HOG_features(image, cell_size=(8, 8), block_size=(3, 3), nbins=9, debug=False, equalize=False):
    '''
    Extract HOG features from an image.
    
    Parameters:
        image: numpy.ndarray
            Input image.
        cell_size: tuple
            Size of the cells.
        block_size: tuple
            Size of the blocks.
        nbins: int
            Number of bins.
        debug: bool
            Whether to show the HOG image.
        equalize: bool
            Whether to equalize the image.
    Returns:
        features: numpy.ndarray
    '''
    if equalize:
        image = equalize_image(image)
    features, hog_img = hog(image, orientations=nbins, pixels_per_cell=cell_size,
                              cells_per_block=block_size, block_norm='L2-Hys', visualize=True)
    if debug:
        plt.imshow(hog_img, cmap='gray')
        plt.show()
    return features






