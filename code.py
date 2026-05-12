import tensorflow as tf
from tensorflow.keras import layers
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2

# DATASETS USED
# MNIST subset
# CIFAR10 subset
# Cats vs Dogs subset
# Text dataset
# COCO dataset

# MLP

def MLP():
    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

    x_train=x_train[:10000]
    y_train=y_train[:10000]
    x_test=x_test[:2000]
    y_test=y_test[:2000]

    x_train=x_train/255.0
    x_test=x_test/255.0

    model=tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    h=model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))

    loss,acc=model.evaluate(x_test,y_test)

    print('Accuracy:',acc)
    print(h.history)
    print('Loss:',loss)

    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.legend(['train','val'])
    plt.show()


# CNN

def CNN():
    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

    x_train=x_train[:5000]
    y_train=y_train[:5000]
    x_test=x_test[:1000]
    y_test=y_test[:1000]

    x_train=x_train/255.0
    x_test=x_test/255.0

    model=tf.keras.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    h=model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))

    loss,acc=model.evaluate(x_test,y_test)

    print('Accuracy:',acc)
    print(h.history)
    print('Loss:',loss)

    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.legend(['train','val'])
    plt.show()


# Transfer Learning

def TransferLearn():
    base=tf.keras.applications.MobileNetV2(weights='imagenet',include_top=False,input_shape=(128,128,3))

    dataset=tf.keras.utils.image_dataset_from_directory(
        'cats_and_dogs_filtered',
        image_size=(128,128),
        batch_size=32
    )

    base.trainable=False

    model=tf.keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(2,activation='softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    print(model.summary())


# CNN vs Transfer Learning

def CNNvsTL():

    import tensorflow as tf
    from tensorflow.keras import layers
    import matplotlib.pyplot as plt

    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

    x_train=x_train[:5000]/255.0
    y_train=y_train[:5000]

    x_test=x_test[:1000]/255.0
    y_test=y_test[:1000]

    # CNN MODEL
    cnn=tf.keras.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])

    cnn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    h1=cnn.fit(
        x_train,
        y_train,
        epochs=3,
        validation_data=(x_test,y_test)
    )

    cnn_loss,cnn_acc=cnn.evaluate(x_test,y_test)

    # TRANSFER LEARNING MODEL
    base=tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(32,32,3)
    )

    base.trainable=False

    tl=tf.keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(10,activation='softmax')
    ])

    tl.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    h2=tl.fit(
        x_train,
        y_train,
        epochs=3,
        validation_data=(x_test,y_test)
    )

    tl_loss,tl_acc=tl.evaluate(x_test,y_test)

    print("CNN Accuracy:",cnn_acc)
    print("Transfer Learning Accuracy:",tl_acc)

    plt.plot(h1.history['accuracy'])
    plt.plot(h2.history['accuracy'])

    plt.legend(['CNN','Transfer'])
    plt.show()


# Performance Analysis

def Analysis():
    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

    x_train=x_train[:10000]
    y_train=y_train[:10000]
    x_test=x_test[:2000]
    y_test=y_test[:2000]

    x_train=x_train/255.0
    x_test=x_test/255.0

    model=tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    h=model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))

    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.legend(['train','val'])
    plt.show()


# Object Detection

def ObjectDetection():
    model=YOLO('yolov8n.pt')

    results=model('test.jpg')

    names=model.names

    for box in results[0].boxes:
        cls=int(box.cls[0])
        conf=float(box.conf[0])

        print('Object:',names[cls])
        print('Confidence:',conf)


# Simple RNN

def RNN():

    import tensorflow as tf
    from tensorflow.keras import layers
    import numpy as np

    text='deep learning lab exam tensorflow rnn lstm gru model prediction'*20

    chars=sorted(list(set(text)))
    char_to_int=dict((c,i) for i,c in enumerate(chars))

    x=[]
    y=[]

    for i in range(0,100,10):

        seq=text[i:i+10]
        label=text[i+10]

        x.append([char_to_int[c] for c in seq])
        y.append(char_to_int[label])

    x=np.array(x)/float(len(chars))
    y=np.array(y)

    x=np.reshape(x,(len(x),10,1))

    model=tf.keras.Sequential([
        tf.keras.Input(shape=(10,1)),
        layers.SimpleRNN(32),
        layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    h=model.fit(x,y,epochs=5)

    print("Loss:",h.history['loss'][-1])
    print(h.history)


# LSTM

def LSTM():
    text='deep learning lab exam tensorflow rnn lstm gru model prediction' * 20

    chars=sorted(list(set(text)))
    char_to_int=dict((c,i) for i,c in enumerate(chars))

    x=[]
    y=[]

    for i in range(0,100,10):
        seq=text[i:i+10]
        label=text[i+10]
        x.append([char_to_int[c] for c in seq])
        y.append(char_to_int[label])

    x=np.array(x)/float(len(chars))
    x=np.reshape(x,(len(x),10,1))
    y=np.array(y)
    y=np.random.rand(1000,1)

    model=tf.keras.Sequential([
        layers.LSTM(32,input_shape=(10,1)),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam',loss='mse',metrics=['mae'])

    h=model.fit(x,y,epochs=5)

    print('Loss:',h.history['loss'][-1])
    print(h.history)


# GRU

def GRU():
    text='deep learning lab exam tensorflow rnn lstm gru model prediction' * 20

    chars=sorted(list(set(text)))
    char_to_int=dict((c,i) for i,c in enumerate(chars))

    x=[]
    y=[]

    for i in range(0,100,10):
        seq=text[i:i+10]
        label=text[i+10]
        x.append([char_to_int[c] for c in seq])
        y.append(char_to_int[label])

    x=np.array(x)/float(len(chars))
    x=np.reshape(x,(len(x),10,1))
    y=np.array(y)
    y=np.random.rand(1000,1)

    model=tf.keras.Sequential([
        layers.GRU(32,input_shape=(10,1)),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam',loss='mse',metrics=['mae'])

    h=model.fit(x,y,epochs=5)

    print('Loss:',h.history['loss'][-1])
    print(h.history)


# GAN

def GAN():
    generator=tf.keras.Sequential([
        layers.Dense(128,activation='relu',input_shape=(100,)),
        layers.Dense(784,activation='sigmoid')
    ])

    noise=np.random.normal(0,1,(1,100))

    img=generator.predict(noise)

    plt.imshow(img.reshape(28,28),cmap='gray')
    plt.show()


# CALL ANY FUNCTION
# MLP()
