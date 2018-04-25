from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from keras.optimizers import rmsprop as rms
from keras.losses import categorical_crossentropy, mse
import numpy as np
import ibench

def model_v1(model_path=None):

    model = Sequential()

    model.add(Conv2D(32, 3, input_shape=(300, 300, 3), activation='tanh'))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, activation='tanh'))
    model.add(MaxPool2D())
    model.add(Conv2D(128, 3, activation='tanh'))
    model.add(MaxPool2D())
    model.add(Flatten())
    #model.add(Dense(1000, activation='relu'))
    #model.add(Dense(1000, activation='relu'))
    model.add(Dense(6, activation='elu'))

    model.compile(optimizer='rmsprop', loss=mse, metrics=['accuracy'])

    if model_path != None:
        model.load_weights(model_path)

    return model

def model_v2(model_path=None):

    model = Sequential()

    model.add(Conv2D(32, 3, input_shape=(150, 150, 3), activation='elu'))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, activation='elu'))
    model.add(MaxPool2D())
    model.add(Conv2D(128, 3, activation='elu'))
    model.add(MaxPool2D())
    model.add(Conv2D(128, 3, activation='elu'))
    model.add(MaxPool2D())
    
    model.add(Flatten())
    model.add(Dense(500, activation='elu'))
    model.add(Dense(500, activation='elu'))
    model.add(Dense(6, activation='elu'))

    model.compile(optimizer=rms(decay=.0001), loss=mse, metrics=['accuracy'])
    model.summary()

    if model_path != None:
        model.load_weights(model_path)

    return model

# model = model_v1()

# img = np.array([ibench.to_array(
#     ibench.to_PIL('./images/I_Chinesebook/I_Chinesebook_0.png').resize((320, 240)))
#     ])

# model.fit(img, np.array([[2,2]]))