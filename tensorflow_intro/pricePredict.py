import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

scaler = MinMaxScaler()
data =  pd.read_excel("prices.xlsx")

# print(data)
x_data = data.values[:,1:]
y_data = data.values[:,:1]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = Sequential()

model.add(Dense(4, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(4, activation="relu"))

model.add(Dense(1))

model.compile(optimizer = "adam", loss= "mse")

model.fit(x_train, y_train, epochs=250)

loss = model.history.history["loss"]

# trainLoss = model.evaluate(x_train, y_train)
# testLoss = model.evaluate(x_test, y_test)

# print(trainLoss)
# print(testLoss)

preds = model.predict(x_test)

tahminDF = pd.DataFrame(y_test, columns=["gercek_degerler"])
tahminDF["tahmin_degerler"] = preds

MAE = mean_absolute_error(y_test, preds)
MSE = mean_squared_error(y_test, preds)

print(tahminDF)
print(MAE)
print(MSE)

# sbn.scatterplot(x ="gercek_degerler", y ="tahmin_degerler", data=tahminDF)

# # yeni = [[13.5, 30.1, -37.4],]
# # yeni = scaler.transform(yeni)

# # yeniT = model.predict(yeni)

# # print(yeniT)

model.save("tahmin_modeli_Sequential.h5")
plt.show()


