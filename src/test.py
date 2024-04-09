
import process as pc
import cv2
import sys
import os

positive_images_path = pc.DATA_PATH+"/img_align_celeba"

def test_extract_dataset():
    pc.extract_dataset(positive_images_path)
    assert "img_align_celeba" in os.listdir(pc.DATA_PATH)
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
        cv2.setMouseCallback("Image", pc.handle_mouse)
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
            images.append(pc.process_image(pc.DATA_PATH+"/img_align_celeba/"+img, resize=True, img_resize=(64, 64)))      
        print(images[0].shape)
        print(images[1].shape)
        print(images[2].shape)
        print("Done 2")

    if sys.argv[1] == "extract_features":
        import numpy as np
        img_list = pc.load_images(pc.DATA_PATH+"/img_align_celeba/", number_of_images=1, random_seed=7)
        images = []
        for img in img_list:
            images.append(cv2.imread(pc.DATA_PATH+"/img_align_celeba/"+img, cv2.IMREAD_GRAYSCALE))
        

        des_append = []
        des_extend = []
        for img_file in img_list:
            img = pc.process_image(pc.DATA_PATH+"/img_align_celeba/"+img_file, resize=True, img_resize=(96, 96))
            _, des = pc.extract_ORB_features(img, n_keypoints=32)
            des_append.append(des)
            des_extend.extend(des)
        
        
        print(len(des_append))
        print(len(des_extend))
        print(des_append[0:2])
        print(des_extend[0:2])

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        print("Done 3")

    if sys.argv[1] == "test_pipeline":
        import joblib
        from matplotlib import image as mpimg
        #pipeline_save_path = pc.DATA_PATH+"/svm_model_3.pkl"
        pipeline_save_path = pc.DATA_PATH+"/svm_model_dog.pkl"
        
        
        image_path = pc.DATA_PATH+"/final/totti_del_piero.jpg"

        pipeline = joblib.load(pipeline_save_path)

        svm = pipeline.named_steps['svc']

        image = cv2.imread(image_path)

        #image = cv2.resize(image, (96, 96))
        faces = pc.detect_faces(image_path, pipeline, threshold=0.65, window_size=(96, 96), step_size=(16,16), n_keypoints=32)    

        for x, y in faces:
            size = 10
            cv2.circle(image, (int(x), int(y)), size, (0, 255, 0), 2)
        
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()