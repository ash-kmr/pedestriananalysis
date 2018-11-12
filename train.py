import numpy as np
import cv2
import os
import sys
import keras
from keras.applications.mobilenet import MobileNet
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.layers as L
from keras.optimizers import SGD, Adam
from os.path import join
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
import matplotlib.pyplot as plt
def pretrained_model():
    model = MobileNet(input_shape=(128, 128, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=7)
    layer_name = 'conv_pw_6'
    intermodel = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermodel

def pretrained_small():
    model = MobileNet(input_shape=(128, 128, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=7)
    layer_name = 'conv_pw_2'
    intermodel = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermodel


def create_PMout():
    train_dir = 'dataset/'
    nTrain = 2472
    train_features = np.zeros(shape=(nTrain, 8, 8, 512))
    train_labels = np.zeros(shape=(nTrain,8))
    print('loading pretrained model')
    datagen = ImageDataGenerator(horizontal_flip=False, vertical_flip=False)
    batchsize = 20
    intermodel = pretrained_model()
    print('pretrained model loaded... creating train_generator')
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batchsize,
        class_mode='categorical')
    i = 0
    print('train_generator created.. getting pretrained model outputs...')
    for inputs_batch, labels_batch in train_generator:
        print('running for batch : '+str(i))
        features_batch = intermodel.predict(inputs_batch)
        train_features[i*batchsize:(i+1)*batchsize] = features_batch
        train_labels[i*batchsize:(i+1)*batchsize] = labels_batch
        i+=1
        if i*batchsize>=nTrain:
            break
    train_features = np.reshape(train_features, (nTrain, 8 * 8 * 512))
    print(train_features, train_labels)
    np.save('PMout/X.npy', train_features)
    np.save('PMout/Y.npy', train_labels)

def getKerasModel():
    model = Sequential()
    #model.add(L.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(8, 8, 512)))
    #model.add(L.Flatten())
    model.add(L.Dense(1000, activation='elu', input_dim=8*8*512))
    model.add(L.Dropout(0.5))
    model.add(L.Dense(128, activation='elu'))
    model.add(L.Dense(8, activation='softmax'))
    return model


def normalize(X):
    for i in range(X.shape[1]):
        maxelem = X[:, i].max()
        X[:, i] = (X[:, i]-maxelem)/maxelem

    return X

def create_data(X, Y):
    data = np.c_[X, Y]
    np.random.shuffle(data)
    train = data[:int(data.shape[0]*0.8), :]
    val = data[int(data.shape[0]*0.8):, :]
    Xtrain = train[:, :-8]
    Ytrain = train[:, -8:]
    Xval = val[:, :-8]
    Yval = val[:, -8:]
    return Xtrain, Ytrain, Xval, Yval


def shuffledata(X, Y):
    data = np.c_[X, Y]
    np.random.shuffle(data)
    X = data[:, :-8]
    Y = data[:, -8:]
    return X, Y

def trainFakeTransfer():
    X = np.load('PMout/X.npy')
    print(X.shape)
    Y = np.load('PMout/Y.npy')
    X, Y = shuffledata(X, Y)
    model = getKerasModel()
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(X, Y, validation_split=0.2, epochs=100, batch_size=64, shuffle=True, verbose=1)


def trainRealTransfer1():
    # working!!!!!!
    batchsize = 64
    datagen = ImageDataGenerator(horizontal_flip=False, vertical_flip=False)
    testgen = ImageDataGenerator(horizontal_flip=False, vertical_flip=False)
    train_generator = datagen.flow_from_directory(
        'dataset/',
        target_size=(128, 128),
        batch_size=batchsize,
        class_mode='categorical')
    test_generator = datagen.flow_from_directory(
        'dataset2/',
        target_size=(128, 128),
        batch_size=batchsize,
        class_mode='categorical')
    intermodel = pretrained_model()
    x = intermodel.output
    x = L.Flatten()(x)
    x = L.Dense(1000, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(128, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(8, activation='softmax')(x)
    model = Model(inputs = intermodel.input, outputs = x)
    model.compile(optimizer=Adam(lr=0.00001, decay=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_generator, steps_per_epoch=40, epochs=300, verbose=1, validation_data=test_generator, validation_steps=48)


def trainRealTransfer2():
    batchsize = 64
    datagen = ImageDataGenerator(horizontal_flip=False, vertical_flip=False)
    testgen = ImageDataGenerator(horizontal_flip=False, vertical_flip=False)
    train_generator = datagen.flow_from_directory(
        'dataset/',
        target_size=(128, 128),
        batch_size=batchsize,
        class_mode='categorical')
    test_generator = datagen.flow_from_directory(
        'dataset2/',
        target_size=(128, 128),
        batch_size=batchsize,
        class_mode='categorical')
    intermodel = pretrained_model()
    x = intermodel.output
    x = L.Flatten()(x)
    x = L.Dense(1000, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(512, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(128, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(8, activation='softmax')(x)
    model = Model(inputs = intermodel.input, outputs = x)
    filepath="models/fullmodel_transfer2.h5"
    model.compile(optimizer=Adam(lr=0.00005, decay=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    history = model.fit_generator(train_generator, steps_per_epoch=40, epochs=40, verbose=1, validation_data=test_generator, validation_steps=48, callbacks=callback_list)
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('models/fullmodel_transfer2_history/accuracy.jpg')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('models/fullmodel_transfer2_history/loss.jpg')





#trainFakeTransfer()
#trainRealTransfer2()




def trainRealTransfer1_small():
    # working!!!!!!
    batchsize = 64
    datagen = ImageDataGenerator(horizontal_flip=False, vertical_flip=False)
    testgen = ImageDataGenerator(horizontal_flip=False, vertical_flip=False)
    train_generator = datagen.flow_from_directory(
        'dataset/',
        target_size=(128, 128),
        batch_size=batchsize,
        class_mode='categorical')
    test_generator = datagen.flow_from_directory(
        'dataset2/',
        target_size=(128, 128),
        batch_size=batchsize,
        class_mode='categorical')
    intermodel = pretrained_small()
    x = intermodel.output
    x = L.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='elu')(x)
    x = L.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='elu')(x)
    x = L.Flatten()(x)
    x = L.Dense(1000, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(128, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(8, activation='softmax')(x)
    filepath='models/fullmodel_transfer_small_pw2.h5'
    model = Model(inputs = intermodel.input, outputs = x)
    print(model.summary())
    model.compile(optimizer=Adam(lr=0.00001, decay=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    history=model.fit_generator(train_generator, steps_per_epoch=40, epochs=50, verbose=1, validation_data=test_generator, validation_steps=48, callbacks=callback_list)
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('models/fullmodel_transfer_small_pw2_history/accuracy.jpg')
    plt.gcf().clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('models/fullmodel_transfer_small_pw2_history/loss.jpg')


def shuffle_in_unison_scary(X, Y):
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)
    return X, Y


def trainRealTransfer2_withoutgen():
    batchsize = 64
    X, Y= getdata()
    X, Y= shuffle_in_unison_scary(X, Y)
    print(Y)
    intermodel = pretrained_model()
    x = intermodel.output
    x = L.Flatten()(x)
    x = L.Dense(1000, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(512, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(128, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(8, activation='softmax')(x)
    model = Model(inputs = intermodel.input, outputs = x)
    filepath="models/fullmodel_transfer2.h5"
    model.compile(optimizer=Adam(lr=0.00005, decay=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x=X, y=Y, batch_size=64, epochs=60, verbose=1, validation_split=0.4)
    
def trainRealTransfer_small_KD():


# custom dataset generation 
def getdata():
    mydir = 'data/extracted/'
    X = np.zeros((2551, 128, 128, 3))
    Y = np.zeros((2551,), dtype=int)
    counter = 0
    for direcs in os.listdir(mydir):
        subpath = os.path.join(mydir, direcs)
        print(direcs)
        for imagename in os.listdir(subpath):
            imagepath = os.path.join(subpath, imagename)
            img = cv2.imread(imagepath)
            img = cv2.resize(img, (128, 128))
            X[counter, :, :, :] = img
            Y[counter] = int(direcs)
            counter+=1
    print('done')
    enc=LabelBinarizer()
    Y=enc.fit_transform(Y.reshape(Y.shape[0], 1))
    return X, Y


trainRealTransfer2_withoutgen()()