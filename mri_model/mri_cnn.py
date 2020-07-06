"""
3D Conv Net for MRI only
"""
import multiprocessing as mp
import pickle
import os
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv3D, Flatten, Dense, MaxPool3D, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from CNNDataGenerator import DataGenerator


def cnn_arch(input_shape):
    model = Sequential()
    model.add(Conv3D(filters=8,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same',
                     input_shape=input_shape,
                     data_format='channels_last',
                     name='input'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=8,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
    model.add(Conv3D(filters=16,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=16,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
    model.add(Conv3D(filters=32,
                     kernel_size=(3, 3, 3),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=32,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=32,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
    model.add(Conv3D(filters=64,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=64,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=64,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
    model.add(Conv3D(filters=128,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=128,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=128,
                     kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool3D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(3, activation='softmax', name='prediction'))

    return model

if __name__ == '__main__':
    params = {'batch_size': 1,
              'dim': (182, 218,  182),
              'n_channels': 1,
              'n_classes': 3,
              'shuffle': True
              }
    epochs = 200
    use_mp = True
    workers = mp.cpu_count()
    num_training = 815

    partition = np.arange(0, num_training)

    IDs = os.listdir("/home/ubuntu/training/MRI/")

    num_to_label = {}

    for subj in IDs:
        num = 0
        if 'AD' in subj:
            num = 0
        elif 'CN' in subj:
            num = 1
        elif 'EMCI' in subj:
            num = 2
        
        num_to_label[subj] = num

    partition = np.arange(0, len(IDs))

    X_train, X_test, _, _ = train_test_split(IDs,
                                             IDs,
                                             test_size=0.2,
                                             random_state=42)

    X_val, X_test, _, _ = train_test_split(X_test,
                                           X_test,
                                           test_size=0.5,
                                           random_state=42)

    training_generator = DataGenerator(X_train, num_to_label, **params)
    validation_generator = DataGenerator(X_val, num_to_label, **params)
    evaluation_generator = DataGenerator(X_test, num_to_label, **params)

    checkpointer = ModelCheckpoint(filepath='bestmodel_cnn.h5',
                                   monitor='val_loss',
                                   save_best_only=True,
                                   verbose=1)

    early_stopper = EarlyStopping(monitor='val_loss',
                                  patience=10,
                                  verbose=1)

    model = cnn_arch((*params['dim'], params['n_channels']))
    print('Compiling model')
    custom_opt = Adam(lr=0.000027, clipvalue=0.5)
    model.compile(optimizer=custom_opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    plot_model(model, to_file='cnn_arch.png', show_shapes=True)

    train_history = model.fit_generator(generator=training_generator,
                                        validation_data=validation_generator,
                                        epochs=epochs,
                                        use_multiprocessing=use_mp,
                                        verbose=1,
                                        callbacks=[checkpointer, early_stopper],
                                        workers=8)

    model = cnn_arch((*params['dim'], params['n_channels']))
    model.load_weights('bestmodel_cnn.h5')
    model.compile(optimizer=custom_opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    tresults = model.evaluate_generator(generator=evaluation_generator,
                                        workers=8,
                                        use_multiprocessing=use_mp,
                                        verbose=1)

    print('Overall accuracy: ' + str(tresults[1]))

    with open('./training_dump_cnn.txt', 'wb') as fp:
        pickle.dump(train_history.history, fp, protocol=-1)