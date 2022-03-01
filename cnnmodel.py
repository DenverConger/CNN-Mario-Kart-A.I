'''
written by Denver Conger

conda activate SAI
pip install tensorflow
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install scikit-learn
'''
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sn
from tensorflow.keras.optimizers import SGD
from keras.layers import LeakyReLU


import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

# Do not INclude this is using it on colab!
# config = tf2.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf2.InteractiveSession(config=config)
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.set_logical_device_configuration(
    physical_devices[0],
    tf.config.LogicalDeviceConfiguration(memory_limit=4096))

  logical_devices = tf.config.list_logical_devices('GPU')
  assert len(logical_devices) == len(physical_devices) + 1

  tf.config.set_logical_device_configuration(
    physical_devices[0],
    tf.config.LogicalDeviceConfiguration(memory_limit=4096))
except:
  pass



# I am trying out the stochastic gradient decent optimizer
opt = SGD(lr=0.001, momentum=0.9)







adam = tf.keras.optimizers.Adam(
    learning_rate= 0.02,
    # epsilon=1e-08,
    name="adam"
)

# Here I am using a predefined model for quicker results. I end up shaving off the last two layers of their massive model and adding a few that I can train from therir previous layers.
def define_model():
    
	model = VGG16(include_top=False, input_shape=(200, 200, 3))

	for layer in model.layers:
		layer.trainable = False

	flattener = Flatten()(model.layers[-1].output)
	layer1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flattener)
	output = Dense(3, activation='softmax')(layer1)

	model = Model(inputs=model.inputs, outputs=output)

    # Currently using precision (which cares more about which ones it got wrong) instead of accuracy
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics = [tf.keras.metrics.Precision()])
	return model


    # model = tf.keras.models.Sequential([
    #     # This is the first convolution
    #     tf.keras.layers.Conv2D(64, (3,3), activation=LeakyReLU(alpha=0.05), input_shape=(200,200,3)),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     # The second convolution
    #     tf.keras.layers.Conv2D(64, (3,3), activation=LeakyReLU(alpha=0.05)),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     # The third convolution
    #     tf.keras.layers.Conv2D(128, (3,3), activation=LeakyReLU(alpha=0.05)),
    #     tf.keras.layers.MaxPooling2D(2,2),
    #     # The fourth convolution
    #     tf.keras.layers.Conv2D(128, (3,3), activation=LeakyReLU(alpha=0.05)),
    #     tf.keras.layers.MaxPooling2D(2,2),
    #     # Flatten the results to feed into a DNN
    #     tf.keras.layers.Flatten(),
    #     # tf.keras.layers.Dropout(0.3),
    #     # 512 neuron hidden layer
    #     tf.keras.layers.Dense(512, activation=LeakyReLU(alpha=0.05)),
    #     tf.keras.layers.Dense(43, activation='softmax')
    # ])

	# model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
	# return model
import sys
def run():
	# define model
    try:
        m = sys.argv.index('-model') + 1
        model_path = sys.argv[m]
    except:
        model_path = None
    try:
        e = sys.argv.index('-epochs')+1
        epochs = int(sys.argv[e])
    except:
        epochs = 5
    try:
        ti = sys.argv.index('-training')+1
        training_dir = f'{dir_path}\\'+sys.argv[ti]
    except:
        training_dir = f'{dir_path}\\training'

    if model_path is None:
        model = define_model()
        model_path = 'kart_model'
    else:
        model = tf.keras.models.load_model(f'{dir_path}\\'+model_path)
        # create data generator
    print("train_dir")
    # training_dir = 'D:\Coding\AIsociety\Mario_Kart\\training'
    image_size = (200, 200)

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=.2,
            zoom_range=.2,
            rotation_range = 40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            )
    validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=.2,
            rotation_range = 40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            )

    train_generator = train_datagen.flow_from_directory(
            training_dir,
            target_size = image_size,
            subset="training",
            batch_size=32,
            class_mode='categorical',
            seed=42,shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
            training_dir,
            target_size=image_size,
            batch_size=32,
            class_mode='categorical',
            subset="validation",
            seed=42)
    print("test_dir")
    # test_dir = 'D:\Coding\AIsociety\Mario_Kart'
    # test_dir = 'C:\Users\dmull\Desktop\Programming_New\AI\Mario\SetUp\simple_kart_ai\simple_kart_ai'
    test_dir = os.path.dirname(os.path.realpath(__file__))
    print(test_dir)


    test_datagen = ImageDataGenerator(rescale=1./255,zoom_range=.2,
            rotation_range = 40,
            width_shift_range=0.2,
            height_shift_range=0.2)
    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(200, 200),
            classes=['test'],
            class_mode='categorical',
            shuffle=False)
        # fit model
    history = model.fit(train_generator, steps_per_epoch=len(train_generator),
        # validation_data=validation_generator, validation_steps=len(validation_generator), epochs=50 , verbose=1)
        validation_data=validation_generator, validation_steps=len(validation_generator), epochs=epochs , verbose=1)
    # evaluate model
    model.save(model_path)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # base = pd.read_csv("D:\Coding\AIsociety\Mario_Kart\\test_classes.csv")
    base = pd.read_csv(f"{dir_path}\\test_classes.csv")
    print(base.head)
    base = base.ClassId
    print(type(base))
    print(type(base[0]))

    pred = model.predict(test_generator, verbose=1)

    cl = []
    print(pred[0])
    for i in range(0, len(pred)):
        cl.append(np.argmax(pred[i]))
    cl = np.array(cl)

    for i in range(len(cl)):
        if base[i] != cl[i]:
            print(i)
            print(cl[i],base[i])
    # for i in range(0,802):
    #     if cl[i] != base[i]:
    #         print("number")
    #         print(i)
    #         print("cl")
    #         print(cl[i])
    #         print("base")
    #         print(base[i])
    Accuracy = accuracy_score(base, cl)
    
    print("accuracy on Test:", Accuracy)
    from sklearn import metrics
    print("Accuracy:",metrics.accuracy_score(base, cl))

    cm = confusion_matrix(base,cl)
    df_cm = pd.DataFrame(cm, range(3), range(3))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    # model.save('kart_model')
run()