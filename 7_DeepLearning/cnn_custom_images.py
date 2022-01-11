import numpy as np
import cv2
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import cm, image, pyplot as plt
from utils import plot_image

cat4 = cv2.imread("../../CATS_DOGS/train/CAT/4.jpg")
cat4 = cv2.cvtColor(cat4, cv2.COLOR_BGR2RGB)
plot_image(cat4, "CAT nr 4")
print(cat4.shape)

dog2 = cv2.imread("../../CATS_DOGS/train/DOG/2.jpg")
dog2 = cv2.cvtColor(dog2, cv2.COLOR_BGR2RGB)
plot_image(dog2, "DOG nr 2")
print(dog2.shape)

image_gen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,  # rotate the image 30 degrees
    width_shift_range=0.1,  # Shift the pic width by a max of 10%
    height_shift_range=0.1,  # Shift the pic height by a max of 10%
    # Rescale the image by normalzing it.
    rescale=1/255,
    # Shear means cutting away part of the image (max 20%)
    shear_range=0.2,
    zoom_range=0.2,  # Zoom in by 20% max
    horizontal_flip=True,  # Allo horizontal flipping
    fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
)

cat4_transformed = image_gen.random_transform(cat4)
plot_image(cat4_transformed, "CAT nr 4 Transformed")
dog2_transformed = image_gen.random_transform(dog2)
plot_image(dog2_transformed, "DOG nr 2 Transformed")


# Let's have Keras resize all the images to 150 pixels by 150 pixels once they've been manipulated.
image_shape = (150, 150, 3)

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                              input_shape=image_shape, activation='relu',))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                              input_shape=image_shape, activation='relu',))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                              input_shape=image_shape, activation='relu',))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128))
# Same thing as applying on top on Dense
model.add(keras.layers.Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(keras.layers.Dropout(0.5))

# Last layer, remember its binary, 0=cat , 1=dog
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

train_image_gen = image_gen.flow_from_directory(
    "../../CATS_DOGS/train", target_size=image_shape[:2], batch_size=16, class_mode="binary")
test_image_gen = image_gen.flow_from_directory(
    "../../CATS_DOGS/test", target_size=image_shape[:2], batch_size=16, class_mode="binary")

print(train_image_gen.class_indices)

# results = model.fit_generator(train_image_gen, epochs=1,
#                               steps_per_epoch=150,
#                               validation_data=test_image_gen,
#                               validation_steps=12)
# print(results.history["acc"])
# plt.plot(results.history['acc'])

# Load the model of 100 epochs that was already generated:
new_model = keras.models.load_model(
    "../../Computer-Vision-with-Python/06-Deep-Learning-Computer-Vision/cat_dog_100epochs.h5")

dog_test = "../../CATS_DOGS/test/DOG/9374.jpg"
dog_test = keras.preprocessing.image.load_img(
    dog_test, target_size=image_shape[:2])
dog_test = keras.preprocessing.image.img_to_array(dog_test)
dog_test = np.expand_dims(dog_test, axis=0)
dog_test = dog_test/255

dog_prediction = new_model.predict(dog_test)
print(f'Probability that image is a dog is: {dog_prediction} ')
dog_prediction = np.round(dog_prediction).astype(int)
print(dog_prediction)


plt.show()
