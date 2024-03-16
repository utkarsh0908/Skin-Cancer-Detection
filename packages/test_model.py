from tensorflow.keras.models import load_model
import pickle

model = load_model("models/SkinCancerDetection3.h5")
      
with open('data/x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
with open('data/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('data/x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)

loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print("Train: accuracy = %f  ;  loss = %f" % (accuracy, loss))
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print("Testing: accuracy = %f  ;  loss = %f" % (accuracy, loss))