# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:11:39 2024

NN for MNIST dataset use tensorflow
"""

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist



(x_train, l_train), (x_test, l_test) = mnist.load_data()


print("first 4 values l_train ", l_train[:4])
print("\n\nlast 4 values l_train ", l_train[-4:])

#print("\n\nfirst  value in x_train \n", x_train[1])

y_train = np.zeros((l_train.shape[0],
                    l_train.max()+1),
                   dtype = np.float32)

y_train[np.arange(l_train.shape[0]), l_train] = 1
y_test = np.zeros((l_test.shape[0],
                   l_test.max() + 1),
                  dtype = np.float32)
y_test[np.arange(l_test.shape[0]), l_test] = 1

#Model 1 : Neurons = 15 , Learning Rate = 0.1 , Batch Size =  16 , Epochs = 30
"""
model_1 = keras.Sequential([
                            keras.layers.Flatten(input_shape = (28, 28)),
                            keras.layers.Dense(15, activation= tf.nn.sigmoid),
                            keras.layers.Dense(10, activation= tf.nn.softmax)            
                            ])

"""
model_1 = keras.Sequential([
                            keras.layers.Flatten(input_shape = (28, 28)),
                            keras.layers.Dense(15, activation= 'sigmoid'),                           
                            keras.layers.Dense(10, activation= 'softmax')
                            ])

model_1.compile(optimizer= tf.keras.optimizers.SGD(0.1),
                loss = 'mean_squared_error',
                metrics = ['accuracy']
                )

model_1.fit(x_train, y_train, epochs = 30, batch_size = 16)

test_loss, test_acc = model_1.evaluate(x_test, y_test)
print("\ntest_accuracy:", test_acc)
print("\ntest_loss:", test_loss)








