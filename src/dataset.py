import os
import zipfile
import numpy as np
import user_warnings as uw
import cv2

zip_file = "real_faces_128.zip"
DATA_PATH = "../data"
def extract_dataset(path: str = None, folder:str = "single_folder"):
    '''
    Extract the dataset from the zip file.

    Parameters:
        path: str
            Path to the dataset.
        folder: str
            Type of folder to extract, either 'single_folder' or 'multi_folders'.
    '''

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
    '''
    Load a random number of images from a folder.
    
    Parameters:
        path: str
            Path to the folder containing the images.
        number_of_images: int
            Number of images to load.
        random_seed: int
            Random seed for reproducibility.
    Returns:
        list of images: list
    '''
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
    '''
    Load an image from a folder.

    Parameters:
        path: str
            Path to the folder containing the image.
        filename: str
            Name of the image file.
    Returns:
        image file name: str
    '''
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
            if(tmp_len >= n_patches):
                break
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



def get_image_list():
    '''
    Get image list from the final folder.

    Returns:
        list of images: list
    '''

    return os.listdir(DATA_PATH+"/final/")

def get_image_path(choice: int):
    '''
    Get the path of an image from the final folder.

    Parameters:
        choice: int
            Index of the image in the list.
    Returns:
        image path: str
    '''
    for i, img in enumerate(get_image_list()):
       if i == choice:
           return DATA_PATH+"/final/"+img