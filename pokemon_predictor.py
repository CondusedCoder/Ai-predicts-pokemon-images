import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf


datadir = "D:\Pictures\pokemon_dataset"
categories = []
img_size = 50

for folder in os.listdir("D:\Pictures\pokemon_dataset"):
    categories.append(folder + "")

categories.remove("dataset")

training_data = []

print("creating data...")

def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        try:

            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([img_array, class_num, os.path.join(path,img)])
                

        except:
            pass


create_training_data()
print("created data")

random.shuffle(training_data)


x = []
y = []
images = []
for features, labels, img in training_data:
    x.append(features)
    y.append(labels)
    images.append(img)

x = np.array(x).reshape(-1, img_size, img_size, 1)
y = np.array(y)

x = x/255.0
print("making the neural network...")

nn = tf.keras.models.Sequential()
nn.add(tf.keras.layers.Flatten())
nn.add(tf.keras.layers.Dense(128, activation="relu"))
nn.add(tf.keras.layers.Dense(128, activation="relu"))
nn.add(tf.keras.layers.Dense(151, activation="softmax"))

print("created the neural network")

print("compiling...")

nn.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

print("copiled")

print("training...")

nn.fit(x,y, epochs=1000)

print("trained")

predictions = nn.predict(x)
for i in range(len(predictions)):

    index = i
    output = np.argmax(predictions[index])
    print(f"prediction: {categories[output]} label: {categories[y[index]]}")
    img = cv2.imread(images[index], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    plt.figure("Image")
    plt.title("image")
    plt.grid(False)

    plt.imshow(img, cmap="gray")
    plt.show()
    input()


