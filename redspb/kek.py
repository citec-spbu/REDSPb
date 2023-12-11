import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

df = pd.read_csv("data_2_test_predict2.csv")
print('1')
df['predict_nn'] = df['predict_nn'].map(lambda x: list(map(lambda x: round(eval(x)[0], 5), x.split(','))))
print('1')
df['predict_nn'] = df['predict_nn'].map(lambda x: str(x)[1:-1])
print('1')
df.to_csv('data_2_test_predict2_.csv', index=False)
# df = pd.read_csv("C:\\Users\\Liza\\Downloads\\data1\\data2.csv")
# # df = pd.read_csv("data_res_2_half_upd.csv")
# df['numbers'] = df['numbers'].map(lambda x: list(map(float, x.split(','))))
# matrix = np.array(df['numbers'].values.tolist(), dtype=float)
# scaler = MaxAbsScaler()
# matrix_numbers = scaler.fit_transform(matrix)
#
# data = pd.read_csv("data_2_test_predict.csv")
# # print(data['predict_nn'].values[0])
# data['predict_nn'] = data['predict_nn'].map(lambda x: list(map(float, x.split(','))))
# matrix_ = np.array(data['predict_nn'].values.tolist(), dtype=float)
#
# def plot_with_slider(data):
#     def plot_threshold(threshold):
#         ax1 = plt.subplot(2, 1, 1)  # 2 строки, 1 столбец, подграфик 1
#         plt.imshow(matrix_numbers, cmap='hot')  # График для первого подграфика
#         thresholded_data = np.where(data > threshold, 1, 0)
#         plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
#         plt.imshow(thresholded_data, cmap='gray')
#         plt.show()
#
#     # Создание окна с ползунком для выбора порогового значения
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.2)
#     threshold_slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])
#     threshold_slider = widgets.Slider(threshold_slider_ax, 'Threshold', 0, 1, valinit=0.5)
#
#     def update(val):
#         plot_threshold(threshold_slider.val)
#
#     threshold_slider.on_changed(update)
#     plt.show()
#
#
# plot_with_slider(matrix_)
