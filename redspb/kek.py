import pandas as pd

df = pd.read_csv("data_2_test_predict+mask.csv")
print('1')
df['predict_nn'] = df['predict_nn'].map(lambda x: list(map(lambda x: round(eval(x)[0], 5), x.split(','))))
print('1')
df['predict_nn'] = df['predict_nn'].map(lambda x: str(x)[1:-1])
print('1')
df.to_csv('data_2_test_predict.csv', index=False)