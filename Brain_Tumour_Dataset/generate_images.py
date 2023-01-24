from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array
import cv2
import os

# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
datagen = ImageDataGenerator(
		rotation_range = 40,
		shear_range = 0.2,
		zoom_range = 0.2,
		horizontal_flip = True,
		brightness_range = (0.5, 1.5))
	
# Loading a sample image
for i in ["no"]:
    collectionImageNames = os.listdir("G:/Brain_Tumour_Dataset" +"/" + str(i))
    for j in collectionImageNames:
        img = cv2.imread("G:/Brain_Tumour_Dataset" + "/" + str(i) + "/" + j)
# Converting the input sample image to an array
x = img_to_array(img)
# Reshaping the input image
x = x.reshape((1, ) + x.shape)

# Generating and saving 5 augmented samples
# using the above defined parameters.
i = 0
for batch in datagen.flow(x, batch_size = 1,
						save_to_dir ='G:/Brain_Tumour_Dataset/generated_dataset/no',
						save_prefix ='image', save_format ='jpeg'):
	i += 1
	if i > 5:
		break
