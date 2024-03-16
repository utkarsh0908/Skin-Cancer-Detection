import tensorflow as tf
import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import packages.model_definition as model_definition


def run_train_model(epochs, train_ds, val_ds): 
  model = model_definition.get_model()
  model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )

  return model