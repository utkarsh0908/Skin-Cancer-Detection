from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

def get_model(input_shape, num_classes):
  model_DenseNet201 = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
  model = Sequential()
  for layer in model_DenseNet201.layers:
    layer.trainable = False
  model.add(model_DenseNet201)
  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(512, activation='relu'))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))

  return model