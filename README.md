## Face Detection System for digital camera :camera: :detective:

### Description
ProCam s.p.a is planning to launch a new compact and inexpensive digital camera for budding photographers.<br>
You will be hired as a Data Scientist to create the system for identifying faces in images, this will then allow 
the photo technicians to optimise the settings for a selfie with one or more people.<br>
This is a computer vision problem, more specifically Face Detection.<br>
You have to provide a scikit-learn pipeline that takes an input image and returns a list with the coordinates of 
bounding boxes where faces are present, if there are no faces in the image the list will obviously be empty.<br>

We selected the following datasets:
- The [Flickr-Faces Dataset (Nvidia) 128x128](https://www.kaggle.com/datasets/dullaz/flickrfaces-dataset-nvidia-128x128) dataset, which contains 70,000 images of faces.
- The [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) dataset, which contains 8,275 images of 101 classes of objects.<br>
- The [Landscape Dataset](https://www.kaggle.com/datasets/arnaud58/landscape-pictures) dataset, which contains 4,319 images of landscapes.

Source code can be found in the src directory, dataset and the final model (model.joblib) are in the data directory.
