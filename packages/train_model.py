import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import pickle
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from livelossplot import PlotLossesKeras
import model_definition
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import augmentor

train_dir = 'data/Train'
test_dir = 'data/Test'

# Create dataframes
train_df = pd.DataFrame(columns=['image_path', 'label'])
test_df = pd.DataFrame(columns=['image_path', 'label'])

# Add images paths and labels to dataframes
for label, directory in enumerate(os.listdir(train_dir)):
    for filename in os.listdir(os.path.join(train_dir, directory)):
      image_path = os.path.join(train_dir, directory, filename)
      train_df = train_df._append({'image_path': image_path, 'label': label}, ignore_index=True)

for label, directory in enumerate(os.listdir(test_dir)):
    for filename in os.listdir(os.path.join(test_dir, directory)):
      image_path = os.path.join(test_dir, directory, filename)
      test_df = test_df._append({'image_path': image_path, 'label': label}, ignore_index=True)
        
# Combine train_df and test_df into one dataframe
df = pd.concat([train_df, test_df], ignore_index=True)
del test_df,train_df

labels = os.listdir(train_dir)

# Create label_map dictionary
label_map = {i: label for i, label in enumerate(labels)}
num_classes=len(label_map)

max_images_per_class = 2500

# Group by label column and take first max_images_per_class rows for each group
df = df.groupby("label").apply(lambda x: x.head(max_images_per_class)).reset_index(drop=True)

import multiprocessing

# Get the number of CPU cores available
max_workers = multiprocessing.cpu_count()

import concurrent.futures

# Define a function to resize image arrays
def resize_image_array(image_path):
  return np.asarray(Image.open(image_path).resize((100,75)))

# Use concurrent.futures to parallelize the resizing process
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Use executor.map to apply the function to each image path in the DataFrame
    image_arrays = list(executor.map(resize_image_array, df['image_path'].tolist()))

# Add the resized image arrays to the DataFrame
df['image'] = image_arrays
del image_arrays

#Augment the resized image arrays
df = augmentor.augmentor(df)

# Use the augmented dataframe for further processing
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

features = df.drop(columns=['label','image_path'], axis=1)
target = df['label']

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20,shuffle=True)

x_train = np.asarray(x_train['image'].tolist())
x_test = np.asarray(x_test['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std

# Perform one-hot encoding on the labels
y_train = to_categorical(y_train,num_classes = num_classes)
y_test = to_categorical(y_test,num_classes = num_classes)

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.20,shuffle=True)
# Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))

y_train = y_train.astype(int)
y_validate = y_validate.astype(int)

input_shape = df['image'][0].shape

with open('data/x_test.pkl', 'wb') as f:
  pickle.dump(x_test, f)
  
with open('data/x_train.pkl', 'wb') as f:
  pickle.dump(x_train, f)

with open('data/y_train.pkl', 'wb') as f:
  pickle.dump(y_train, f)

with open('data/y_test.pkl', 'wb') as f:
  pickle.dump(y_test, f)

model = model_definition.get_model(input_shape, num_classes)

# compile model
from keras.optimizers import SGD
opt = SGD(learning_rate=0.0001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=5,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# Fit the model
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=120,
                    validation_split=0.2,
                    batch_size=32,
                    validation_data=(x_validate,y_validate)
                   )

model.save("models/SkinCancerDetection3.h5")