import numpy as np
import pandas as pd
from keras.src.layers import MaxPooling2D
from matplotlib import widgets
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Flatten, Dense, Conv2DTranspose, Input, Conv1D, InputLayer, BatchNormalization
from keras.models import Model, Sequential
# from keras.optimizers import Adam
from form_dataset import generate_dataset
from tensorflow.keras.optimizers.legacy import Adam

df = pd.read_csv("/Users/aleksandrallahverdan/Downloads/data_res_2_half_upd.csv")
df['numbers'] = df['numbers'].map(lambda x: list(map(float, x[1:-1].split(','))))
df['target'] = df['target'].map(lambda x: list(map(float, x.split(','))))
X, Y, true_raw_shape, scaler = generate_dataset(df, info=True)
del df

def train_test_split(X, Y, test_size=0.2):
    assert 0 < test_size < 1
    n = int(X.shape[0] * (1-test_size))
    return X[:n], X[n:], Y[:n], Y[n:]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

del X
del Y


# def min_max_scaler(arr: np.ndarray):
#     return (arr - arr.min()) / (arr.max() - arr.min())
#
#
# for i in range(len(x_train)):
#     x_train[i] = min_max_scaler(x_train[i])
# for i in range(len(x_test)):
#     x_test[i] = min_max_scaler(x_test[i])

# Архитектура сети
input_shape = (10, 100, 1)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='same'),
    MaxPooling2D((2, 2), strides=1, padding='same'),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=1, padding='same'),
    Conv2D(1, kernel_size=(3, 3), activation='linear', padding='same'),
    # Flatten(),
    # Dense(1000, activation='linear'),
    # Dense(10,  activation='softmax')
])

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
model.fit(x_train, y_train, batch_size=512, epochs=3, validation_split=0.2)

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


def plot_with_slider(data):
    def plot_threshold(threshold):
        ax1 = plt.subplot(2, 1, 1)  # 2 строки, 1 столбец, подграфик 1
        plt.imshow(rec1, cmap='hot')  # График для первого подграфика
        thresholded_data = np.where(data > threshold, 1, 0)
        plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
        plt.imshow(thresholded_data, cmap='gray')
        plt.show()

    # Создание окна с ползунком для выбора порогового значения
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    threshold_slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])
    threshold_slider = widgets.Slider(threshold_slider_ax, 'Threshold', 0, 1, valinit=0.5)

    def update(val):
        plot_threshold(threshold_slider.val)

    threshold_slider.on_changed(update)
    plt.show()


# plt.subplot(2, 1, 1)  # 2 строки, 1 столбец, подграфик 1
# plt.imshow(rec1, cmap='hot')  # График для первого подграфика
# plt.title('Subplot 1')  # Заголовок для первого подграфика
#
# # Создание второго подграфика
# plt.subplot(2, 1, 2)  # 2 строки, 1 столбец, подграфик 2
# plt.imshow(rec2, cmap='hot')  # График для второго подграфика
# plt.title('Subplot 2')  # Заголовок для второго подграфика
#
# # Отображение графиков
# plt.tight_layout()  # Автоматическое расположение подграфиков
# plt.show()

plot_with_slider(rec2)
