import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


# Evaluate if the bank note is valid or a fake
# for this example we just use features from the images not the images itself

data = genfromtxt("../DATA/bank_note_data.txt", delimiter=",")
print(data)

# these data contains features of the images and are already labeled as valid or invalid bank notes:
# 0 - invalid
# 1 - valid
# we will feed the image features to the model and figure out the class (valid or invalid)

labels = data[:, 4]
features = data[:, 0:4]

X = features
y = labels

# Separe the features into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# test_size of 0.33 means that 33% of the data will be in the test set
# random_state to shuffle everytime with the same value

# Do a MinMaxScale to the data. This is important when the min and max values are of different ranges like max:2500.0 and min:-13.0
print(X_train.max())
print(X_train.min())
# in this case they are not but it's always good to do so

scaler_object = MinMaxScaler()

# Finds the min and max
scaler_object.fit(X_train)
# Transform both train and test only with the fitting of the train data otherwise, you're assuming some knowledge of the test data that in real life you're not going to have.
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

# Train the data
model = keras.Sequential()
model.add(layers.Dense(4, input_dim=4, activation="relu"))  # input layer
model.add(layers.Dense(8, activation="relu"))  # hidden layer with 8 neurons
model.add(layers.Dense(1, activation="sigmoid"))  # output layer
model.summary()
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])
model.fit(scaled_X_train, y_train, epochs=50, verbose=2)  # train

# So now we trained the model so how can we now predict on new data?
# We predict on the test set because it never trained on the test set:
predict_x = model.predict(scaled_X_test)
predictions_classes_x = np.round(predict_x).astype(int)
print(predictions_classes_x)
print(model.metrics_names)

print(confusion_matrix(y_test, predictions_classes_x))
print(classification_report(y_test, predictions_classes_x))

model.save("my_bank_note_model.h5")

saved_model = models.load_model("my_bank_note_model.h5")
predict_x = saved_model.predict(scaled_X_test)
predictions_classes_x = np.round(predict_x).astype(int)
# print(predictions_classes_x)
print(confusion_matrix(y_test, predictions_classes_x))
print(classification_report(y_test, predictions_classes_x))
