# Script by Arnault VERRET
#
# Lib used :
# imbalanced-learn 0.9.0
# Keras            2.3.1
# numpy            1.19.5
# opencv-python    4.5.5.64
# pandas           1.4.2
# scikit-learn     1.0.2
# tensorflow       2.2.0
#
#
# An old version of tensorflow was used to be 
# able to use the GPU
#
#
# Hardware :
# cpu : i7-4710HQ
# gpu : GTX 980M
# ram : 16Go DDR3 1600MHz
# ssd : 500 MB/s (for swap memory)



import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input
)

import cv2

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# tensorflow setup
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) < 1:
    print("Warning : no GPU found, the algorithm might take a while...")
else:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# important consts
H, W = 128, 128
path_to_train = "Train/Train/"
path_to_test = "Test/Test/"

def read_image(path):
    """
    Give a cv2 image from a path
    """
    if not isinstance(path, str):
        path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x  

def under_over_sampling(X, Y, max_diff_ratio=1):
    """
    Take a dataset with its labels and compute an under sampling followed by an over sampling
    To match a maximum ratio between the max and min of the labels count
    """

    unique, counts = np.unique(Y, return_counts=True)
    l = dict(zip(unique, counts))

    counts = np.array(counts)
    target_counts = max_diff_ratio * (counts - counts.mean()) / counts.std() + counts.mean()
    target = dict(zip(unique, target_counts.astype(int)))
    print(target)

    rus_d = {}
    for k, v in l.items():
        if v > target[k]:
            rus_d[k] = target[k]
        else:
            rus_d[k] = v
    rus = RandomUnderSampler(rus_d)
    X, Y = rus.fit_resample(X, Y)

    ros_d = {}
    for k, v in rus_d.items():
        if v < target[k]:
            ros_d[k] = target[k]
        else:
            ros_d[k] = v

    ros = RandomOverSampler(ros_d)
    X, Y = ros.fit_resample(X, Y)

    return X, Y

def load_train_data(path):
    """
    Load data from path
    return: (list of cv2 images, labels)
    """
    df = pd.read_csv("metadataTrain.csv")
    print(df["CLASS"].value_counts())
    df = pd.concat(
        [
            df["ID"],
            df["CLASS"],
            df["AGE"],
            df['SEX'].str.get_dummies(),
            df['POSITION'].str.get_dummies()
        ],
        axis=1
    )
    df["AGE"] = df["AGE"].fillna(0)
    df["AGE"] = df["AGE"] / df["AGE"].abs().max()

    Y = (df.to_numpy()[:, 1] - 1).astype(int)
    X = [path_to_train+id+".jpg" for id in df["ID"]] # data path

    X = pd.DataFrame({'col':X})

    # we deal with unbalanced dataset here for training.
    # If the dataset is still unbalanced, the class_weights will
    # manage to counter it.

    X, Y = under_over_sampling(X, Y, max_diff_ratio=1000)

    X, Y = shuffle(X, Y)

    X = np.asarray([read_image(x) for x in X["col"]])

    return X, Y

def load_test_data(path):
    df = pd.read_csv("metadataTest.csv")
    df = pd.concat(
        [
            df["ID"],
            df["AGE"],
            df['SEX'].str.get_dummies(),
            df['POSITION'].str.get_dummies()
        ],
        axis=1
    )
    df["AGE"] = df["AGE"].fillna(0)
    df["AGE"] = df["AGE"] / df["AGE"].abs().max()

    X = [path+id+".jpg" for id in df["ID"]] # data path

    X = np.asarray([read_image(x) for x in X])

    return X, df["ID"]

def get_model():

    input_layer = Input((W, H, 3))


    preprocessed_input = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(input_layer)

    mobile = tf.keras.applications.mobilenet.MobileNet()
    mobile.trainable = True # We begin with the pre trained model. but allowing it to further train on this specific tasks gives greater results.
    x = mobile.layers[-6].output
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(8, activation='softmax')(x)
    model_to_train = Model(inputs=mobile.input, outputs=predictions)

    x = model_to_train(preprocessed_input)
    model = Model(inputs = input_layer, outputs=x)

    return model

def main():
    # load data

    X, Y = load_train_data(path_to_train)

    # Data generation for better learning

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2
    )

    datagen.fit(X)

    # Create model

    model = get_model()

    # train model

    loss_fn = SparseCategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    model.summary()

    unique, counts = np.unique(Y, return_counts=True)
    class_weights = len(counts) * counts / np.sum(counts)

    model.fit(
        datagen.flow(X, Y, batch_size=32, shuffle=True),
        epochs=10,
        class_weight = dict(zip(unique, class_weights))
    )

    # print classification result on each classes

    y_pred = model.predict(X).argmax(axis=-1)
    print(classification_report(Y, y_pred))

    # predict on test dataset

    X_test, ids = load_test_data("Test/Test/")
    Y_pred = model.predict(X_test).argmax(axis=-1)

    pred = pd.DataFrame({'CLASS':Y_pred})

    df = pd.concat(
        [
            ids,
            pred
        ],
        axis=1
    )
    df["CLASS"]+=1
    
    df.to_csv("prediction.csv", index=False)

if __name__ == '__main__':
    main()

