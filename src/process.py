import os
import zipfile
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float
import numpy as np

zip_file = "img_align_celeba.zip"
DATA_PATH = "../data"

def extract_dataset():
    if not os.path.exists(DATA_PATH+"/img_align_celeba"):
        print("Extracting CelebA dataset...")
        with zipfile.ZipFile(DATA_PATH+'/'+zip_file, 'r') as zip_ref:
            zip_ref.extractall(DATA_PATH)
    else:
        print("CelebA dataset already extracted")

def load_images(number_of_images=100, random_seed=42):
    if not os.path.exists(DATA_PATH+"/img_align_celeba"):
        extract_dataset()
    list_of_images = os.listdir(DATA_PATH+"/img_align_celeba")
    np.random.seed(random_seed)
    images = np.random.choice(list_of_images, number_of_images)
    return images

def load_image(filename=None):
    if filename is None:
        if not os.path.exists(DATA_PATH+"/img_align_celeba"):
            extract_dataset()
        list_of_images = os.listdir(DATA_PATH+"/img_align_celeba")
        filename = DATA_PATH+"/img_align_celeba/"+list_of_images[0]
        print("Loaded image: ", filename)
    else:
        if filename not in os.listdir(DATA_PATH+"/img_align_celeba"):
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

def process_image(image_file):
    image = denoise_image(image_file)
    image = cv2.cvtColor(image.astype("float32"), cv2.COLOR_BGR2GRAY)
    return image

def extract_features(image):
    pass
    


# Main function (for debugging)
if __name__ == "__main__":
    extract_dataset()
    img_file = load_image()
    
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", handle_mouse)
    image = denoise_image(img_file)
    image = cv2.cvtColor(image.astype("float32"), cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image", image)

    while True:
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            break
    cv2.destroyAllWindows()
    print("Done 1")

    img_list = load_images(number_of_images=100, random_seed=7)
    images = []
    for img in img_list:
        images.append(process_image(DATA_PATH+"/img_align_celeba/"+img))
    print("Done 2")

    print(images[0].shape)
    print(images[1].shape)
    print(images[2].shape)



