
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from keras.models import Sequential

#loading the data
X_train=np.load('train_data.npy')
X_test=np.load('test_data.npy')

#loading the ladels
y_train=np.load('train_labels.npy')
y_test=np.load('test_labels.npy')

#initializing the model
model=Sequential()

#adding the layers
#1st hidden layer
model.add(Conv2D(filters=4,kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
model.add(AveragePooling2D(pool_size=(2,2)))

#2nd hidden layer
model.add(Conv2D(filters=4, kernel_size=(7,7), activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

#flattening the output
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))

#compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#training the model
model.fit(X_train, y_train, epochs=5)

#testing the model
score=model.evaluate(X_test, y_test)
print('The accuracy of the model is: ', int(score[1])*100)


#saving the model
model.save('model.h5')






