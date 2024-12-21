from keras.datasets import mnist
import numpy as np

import os
#extracting training data and labels
(X_train, y_train),(X_test, y_test)=mnist.load_data()

#reshaping the data
X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)

#save data as binary files
np.save('train_data.npy', X_train)
np.save('test_data.npy', X_test)

#save labels as binary files
np.save('train_labels.npy', y_train)
np.save('test_labels.npy',y_test)
