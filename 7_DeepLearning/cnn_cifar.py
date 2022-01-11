import numpy as np
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import cm, image, pyplot as plt
from utils import plot_image

# Load CIFAR data : contains animals in RGB
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(x_train.shape)  # (50000, 32, 32, 3)
single_image = x_train[0]
plot_image(single_image, "First Image : " +
           str(y_train[0]), cmap_option="gray")

# Convert the y_train to one hot encoding
# 10: because we have 10 categories https://www.cs.toronto.edu/~kriz/cifar.html
nr_categories = 10
y_cat_train = to_categorical(y_train, nr_categories)
y_cat_test = to_categorical(y_test, nr_categories)
print(y_cat_train.shape)  # (60000, 10)

# Create Model
model = keras.models.Sequential()

# Convolution Layer
model.add(keras.layers.Conv2D(filters=32, kernel_size=(4, 4),
                              input_shape=(32, 32, 3), activation="relu"))
# Pooling Layer
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
# 2nd Convolution Layer
model.add(keras.layers.Conv2D(filters=32, kernel_size=(4, 4),
                              input_shape=(32, 32, 3), activation="relu"))
# Pooling Layer
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
# Flatten Layer to go from 2D to 1D since our categories are 1D
model.add(keras.layers.Flatten())
# Dense Layer (128,256,512...)
model.add(keras.layers.Dense(256, activation="relu"))
# Output Layer
model.add(keras.layers.Dense(nr_categories, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop", metrics=["accuracy"])
model.summary()

model.fit(x_train, y_cat_train, epochs=2)
print(model.metrics_names)

# Evaluate on test set
model.evaluate(x_test, y_cat_test)
# If you have a poor accuracy in your test data but really good on the training set that means you're probably overfitting to the training set.

predictions = model.predict(x_test)
print(predictions)
predictions = np.argmax(predictions, axis=1)
print(predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# model.save("cifar_10epochs.h5")

# Using an already trained model
saved_model = keras.models.load_model(
    "../../Computer-Vision-with-Python/06-Deep-Learning-Computer-Vision/cifar_10epochs.h5")
predictions = saved_model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


plt.show()
