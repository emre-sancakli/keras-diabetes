# libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# load the dataset
dataset = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)

# split into input and output labels
dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

# We reserve 80% of the dataset for training and 20% for testing.
training_dataset_x, test_dataset_x,  training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

# define the keras model
model = Sequential()
# adding layers to our keras model
model.add(Dense(100, activation='relu', input_dim=8, name='Hidden-1'))
model.add(Dense(100, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
# it gives a brief overview of our model
model.summary()

# compile the keras model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

# fit the keras model on the dataset
model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=130, validation_split=0.20)

# evaluate the keras model
result = model.evaluate(test_dataset_x, test_dataset_y)
# it gives loss value and metrics value that we used
print(result)


# 4 randomly selected examples from csv file.(excluding the last column values)
predict_data = np.array([[1,114,66,36,200,38.1,0.289,21],
                         [1,147,94,41,0,49.3,0.358,27], 
                         [3,130,78,23,79,28.4,0.323,34], 
                         [2,88,58,26,16,28.4,0.766,22]])

#1,114,66,36,200,38.1,0.289,21,0
#1,147,94,41,0,49.3,0.358,27,1
#3,130,78,23,79,28.4,0.323,34,1
#2,88,58,26,16,28.4,0.766,22,0

# makes predictions with the model
predicted_results = model.predict(predict_data)

# if the predicted_results values greater than 0.5, then the sample is diabetic
for result in predicted_results[:, 0]:
    if result > 0.5:
        print('Diabetic')
    else:
        print('Not Diabetic')




# lines of code for graph drawing. 
# before, you must assign the line with the fit operation to the hist variable
"""
import matplotlib.pyplot as plt

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Loss - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, len(hist.history['loss']) + 1), hist.history['loss'])
plt.plot(range(1, len(hist.history['val_loss']) + 1), hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Binary Accuracy - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Binary Accuracy')
plt.plot(range(1, len(hist.history['binary_accuracy']) + 1), hist.history['binary_accuracy'])
plt.plot(range(1, len(hist.history['val_binary_accuracy']) + 1), hist.history['val_binary_accuracy'])
plt.legend(['Binary Accuracy', 'Validation Binary Accuracy'])
plt.show()
"""


