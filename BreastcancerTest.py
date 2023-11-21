import cv2
from keras.models import load_model
from PIL import Image
import numpy as np


model=load_model('BreastCancerDetectionModel.h5')

image=cv2.imread('D:/Nitin Punia/Core Module 5/MediVision/Datasets/Breastcancer_no/normal (74).png')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict(input_img) # type: ignore
print(result) 