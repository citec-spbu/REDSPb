import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Dropout, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from form_dataset import generate_dataset

path_to_file = "D:\practice\part1.csv"
df = pd.read_csv(path_to_file)
df['numbers'] = df['numbers'].map(lambda x: list(map(float, x[1:-1].split(','))))
df['target'] = df['target'].map(lambda x: list(map(float, x.split(','))))
X, Y, true_raw_shape = generate_dataset(df, info=True)
del df

def train_test_split(X, Y, test_size=0.2):
    assert 0 < test_size < 1
    n = int(X.shape[0] * (1-test_size))
    return X[:n], X[n:], Y[:n], Y[n:]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4)

del X
del Y

# Архитектура сети
input_shape = (10, 100, 1)
inputs = Input(shape=input_shape)

x = Conv2D(32, kernel_size=(5, 5), padding='same')(inputs)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.2)(x)

x = Conv2D(64, kernel_size=(5, 5), padding='same')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.2)(x)

outputs = Conv2D(1, kernel_size=(3, 3), activation='linear', padding='same')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
model.fit(x_train, y_train, batch_size=256, epochs=3, validation_split=0.2)

predicted_Y = model.predict(x_test)

def reconstruct_matrix_super(matrix_target_shape, y, n=10, step_n=10, step_m=10):
    res = []
    i = 0
    while len(y) > matrix_target_shape[1]*(i+1):
        layer = y[matrix_target_shape[1]*i:matrix_target_shape[1]*(i+1):y.shape[2]//step_m]
        layer = np.concatenate(layer, axis=1)
        res.append(layer)
        i += n // step_n
    res = np.concatenate(res, axis=0)
    return res




rec1 = reconstruct_matrix_super(true_raw_shape, x_test)
rec2 = reconstruct_matrix_super(true_raw_shape, predicted_Y)

plt.subplot(2, 1, 1)
plt.imshow(rec1, cmap='hot')
plt.title('Subplot 1')


plt.subplot(2, 1, 2)
plt.imshow(rec2, cmap='hot')
plt.title('Subplot 2')

plt.tight_layout()
plt.show()