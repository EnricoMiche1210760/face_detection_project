
import process as pc
import cv2
import sys
import os
import numpy as np
import dataset as ds


positive_images_path = ds.DATA_PATH+"/real_faces_128"

def test_extract_dataset():
    pc.extract_dataset(positive_images_path)
    assert "real_faces_128" in os.listdir(ds.DATA_PATH)
    print("PASS")

def test_load_image():
    img_file = pc.load_image(positive_images_path)
    assert os.path.exists(img_file)
    print("PASS")
    assert img_file is not None
    print("PASS")


def test_image_denoise():
    img_file = pc.load_image(positive_images_path)
    image = pc.denoise_image(img_file)
    assert image is not None
    print("PASS")
    assert image.shape == (218, 178, 3)
    print("PASS")


def test_image_equalize():
    img_file = pc.load_image(positive_images_path)
    image = pc.denoise_image(img_file)
    eq_image = pc.equalize_image(image)
    assert eq_image is not None
    print("PASS")
    assert eq_image.shape == (218, 178)
    print("PASS")

def handle_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button clicked at: ({}, {})".format(x, y))


# Main function (for debugging)
if __name__ == "__main__":

    if len(sys.argv) == 1:
        test_extract_dataset()
        test_load_image()
        test_image_denoise()
        test_image_equalize()
        sys.exit(0)
   
    if sys.argv[1] == "handle_mouse_test":
        pc.extract_dataset()
        img_file = pc.load_image()
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Image", handle_mouse)
        image = pc.denoise_image(img_file)
        image = cv2.cvtColor(image.astype("float32"), cv2.COLOR_BGR2GRAY)
        cv2.imshow("Image", image)

        while True:
            if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                break
        cv2.destroyAllWindows()
        print("OK")

    if sys.argv[1] == "image_shape":
        img_list = pc.load_images(number_of_images=100, random_seed=7)
        images = []
        for img in img_list:
            image = cv2.imread(ds.DATA_PATH+"/real_faces_128/"+img)
            images.append(pc.process_image(image, resize=True, img_resize=(64, 64)))      
        print(images[0].shape)
        print(images[1].shape)
        print(images[2].shape)
        print("Done 2")

    if sys.argv[1] == "extract_ORB_features":
        img_list = pc.load_images(ds.DATA_PATH+"/real_faces_128/", number_of_images=10, random_seed=7)
        images = []
        for img in img_list:
            images.append(cv2.imread(ds.DATA_PATH+"/real_faces_128/"+img, cv2.IMREAD_GRAYSCALE))

        des_append = []
        des_extend = []
        for img_file in img_list:
            image = cv2.imread(ds.DATA_PATH+"/real_faces_128/"+img_file)
            img = pc.process_image(image, resize=True, img_resize=(128, 128))
            _, des = pc.extract_ORB_features(img, debug=True)
        
        print("Done 3")

    if sys.argv[1] == "extract_SIFT_features":
        img_list = pc.load_images(ds.DATA_PATH+"/real_faces_128/", number_of_images=10, random_seed=7)
        images = []
        for img in img_list:
            images.append(cv2.imread(ds.DATA_PATH+"/real_faces_128/"+img, cv2.IMREAD_GRAYSCALE))

        des_append = []
        des_extend = []
        for img_file in img_list:
            image = cv2.imread(ds.DATA_PATH+"/real_faces_128/"+img_file)
            img = pc.process_image(image, resize=True, img_resize=(128, 128))
            kp, des = pc.extract_SIFT_features(img, debug=True)

        print("Done 4")


    if sys.argv[1] == "test_SIFT_pipeline":
        import joblib
        from matplotlib import image as mpimg
        pipeline_save_path = ds.DATA_PATH+"/sift_features_32.pkl"
        image_path = ds.DATA_PATH+"/final/Valentino_Rossi_2017.jpg"

        pipeline = joblib.load(pipeline_save_path)

        svm = pipeline.named_steps['svc']

        image = cv2.imread(image_path)
        print(image.shape)
        height = image.shape[0]/2
        width = image.shape[1]/2
        image = cv2.resize(image, (int(width), int(height)))
        print(image.shape)

        try:
            faces, _ = pc.detect_faces(image, pipeline, method='SIFT', threshold=0.65, window_size=(128, 128), step_size=(64, 64), n_keypoints=32, resize=False)

            for x, y in faces:
                size = 10
                cv2.circle(image, (int(x), int(y)), size, (0, 255, 0), 2)
        except:
            print("No face detected")

        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if sys.argv[1] == "test_ORB_pipeline":
        import joblib
        from matplotlib import image as mpimg
        pipeline_save_path = ds.DATA_PATH+"/svm_model_3.pkl"
        image_path = ds.DATA_PATH+"/final/Valentino_Rossi_2017.jpg"

        pipeline = joblib.load(pipeline_save_path)

        svm = pipeline.named_steps['svc']

        image = cv2.imread(image_path)
        print(image.shape)
        height = image.shape[0]/2
        width = image.shape[1]/2
        image = cv2.resize(image, (int(width), int(height)))
        print(image.shape)

        try:
            faces, _ = pc.detect_faces(image, pipeline, threshold=0.5, method='ORB', window_size=(128, 128), step_size=(64, 64), n_keypoints=40, resize=False)
            for x, y in faces:
                size = 10
                cv2.circle(image, (int(x), int(y)), size, (0, 255, 0), 2)
        except:
            print("No face detected")

        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()