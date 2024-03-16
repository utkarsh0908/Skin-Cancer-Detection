import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def augmentor(df):
    max_images_per_class = 2500

    # Create an ImageDataGenerator object with the desired transformations
    datagen = ImageDataGenerator(
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

    augmented_df = pd.DataFrame(columns=['image_path', 'label', 'image'])

    # Loop through each class label and generate additional images if needed
    for class_label in df['label'].unique():
      # Get the image arrays for the current class
      image_arrays = df.loc[df['label'] == class_label, 'image'].values

      # Calculate the number of additional images needed for the current class
      num_images_needed = max_images_per_class - len(image_arrays)

      # Generate augmented images for the current class
      if num_images_needed > 0:
          # Select a random subset of the original images
          selected_images = np.random.choice(image_arrays, size=num_images_needed)

          # Apply transformations to the selected images and add them to the augmented dataframe
          for image_array in selected_images:
              # Reshape the image array to a 4D tensor with a batch size of 1
              image_tensor = np.expand_dims(image_array, axis=0)

              # Generate the augmented images
              augmented_images = datagen.flow(image_tensor, batch_size=1)

              # Extract the augmented image arrays and add them to the augmented dataframe
              for i in range(augmented_images.n):
                  augmented_image_array = augmented_images.next()[0].astype('uint8')
                  augmented_df = augmented_df._append({'image_path': None, 'label': class_label, 'image': augmented_image_array}, ignore_index=True)

      # Add the original images for the current class to the augmented dataframe
      original_images_df = df.loc[df['label'] == class_label, ['image_path', 'label', 'image']]
      augmented_df = augmented_df._append(original_images_df, ignore_index=True)

    # Group the augmented dataframe by the 'label' column and filter out extra images
    df = augmented_df.groupby('label').head(max_images_per_class)

    del augmented_df
    return df