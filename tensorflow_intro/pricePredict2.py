import os
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


raw_data = pd.read_csv("merc.csv")
# print(raw_data.describe())

# dropping non numeric attrs
raw_data.drop(["model","fuelType","transmission"], inplace=True, axis=1)

# cleaning outliers by price
raw_data = raw_data.sort_values("price",ascending=True)
q1 = raw_data["price"].quantile(0.25)
q3 = raw_data["price"].quantile(0.75)
iqr = q3 - q1
iqrMin = q1 - (1.5*iqr)
iqrMax = q3 + (1.5*iqr)
clean_data = raw_data[~((raw_data["price"]<iqrMin) | (raw_data["price"]>iqrMax))]

# sbn.displot(raw_data["price"])
# sbn.displot(clean_data["price"])
# sbn.boxplot(raw_data["price"])
# print(raw_data.info())
# sbn.scatterplot(x="mileage", y="price", data = raw_data, )

# print(clean_data.groupby("year").mean()["price"])
# print(clean_data.sort_values("year", ascending= True))

clean_data = clean_data[clean_data["year"] != 1970]

x_data = clean_data.drop("price", axis=1).values
y_data = clean_data["price"].values

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
sscaler = MinMaxScaler()
sscaler = sscaler.fit(x_train)

x_test = sscaler.transform(x_test)
x_train = sscaler.transform(x_train)

# modelling
model = Sequential()
model.add(Dense(6,activation="relu"))
model.add(Dense(6,activation="relu"))
model.add(Dense(6,activation="relu"))
model.add(Dense(6,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

model.fit(x= x_train, y= y_train, batch_size=32 , validation_data=(x_test,y_test), epochs=250)
loss_data = pd.DataFrame(model.history.history)

preds = model.predict(x_test)
predictDf = pd.DataFrame(preds, columns=["predictions"])
predictDf["trueValues"] = y_test

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)

print(predictDf)
print(f"""
MAE: {mae},
MSE: {mse}      
""")

# model.save("sequential_model_2.h5")

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

sbn.scatterplot(x="predictions", y="trueValues", data=predictDf, ax=axes[0])
axes[0].set_title("Predictions vs True Values")
axes[0].set_xlabel("Predictions")
axes[0].set_ylabel("True Values")

loss_data.plot(ax=axes[1])
axes[1].set_title("Model Loss Over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")


plt.show()