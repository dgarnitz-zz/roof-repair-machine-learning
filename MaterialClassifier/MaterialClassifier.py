#This code was adopted from a template provided by the Udemy Advanced Computer Vision
#https://github.com/lazyprogrammer/machine_learning_examples/tree/master/cnn_class2

from __future__ import print_function, division
from builtins import range, input

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob


# re-size all the images to this
IMAGE_SIZE = [200, 200] 

# training config - discovered this as optimal through trial and error
epochs = 100
batch_size = 20

#set paths to training and validation data
train_path = './images/training'
valid_path = './images/validation'

# useful for getting number of files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# useful for getting number of classes
folders = glob(train_path + '/*')

# add preprocessing layer to the front of VGG
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in resnet.layers:
  layer.trainable = False

# add additional dense layers layers 
x = Flatten()(resnet.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)


# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=["acc"]
)



# create an instance of ImageDataGenerator
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)


# get label mapping for confusion matrix plot later
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
  labels[v] = k


# create generators
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)


# fit the model
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)



def get_confusion_matrix(data_path, N):
  # we need to see the data in the same order
  # for both predictions and targets
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
    i += 1
    if i % 50 == 0:
      print(i)
    p = model.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm

#training confusion matrix
cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
# sns.heatmap(cm, annot=True, xticklabels=["Asphalt", "Clay", "Metal", "Wood"], yticklabels=["Asphalt", "Clay", "Metal", "Wood"], fmt='.0f')
# plt.title("Material Classifier Training Confusion Matrix")
# plt.savefig("material-training-confusion-matrix.png")

#testing confusion matrix
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)
# sns.heatmap(valid_cm, annot=True, xticklabels=["Asphalt", "Clay", "Metal", "Wood"], yticklabels=["Asphalt", "Clay", "Metal", "Wood"], fmt='.0f')
# plt.title("Material Classifier Testing Confusion Matrix")
# plt.savefig("material-testing-confusion-matrix.png")


# plot the results

# loss
# plt.plot(r.history['loss'], label='train loss')
# plt.plot(r.history['val_loss'], label='val loss')
# plt.legend()
# #plt.show()
# plt.savefig('loss-matclass.png')

# accuracies
# plt.plot(r.history['acc'], label='train acc')
# plt.plot(r.history['val_acc'], label='val acc')
# plt.legend()
# #plt.show()
# plt.savefig('accuracies-matclass.png')


# save the model to disk
# model.save('/cs/home/dag8/Documents/Dissertation/Code/API/material_classifier/ResNet50.h5')
