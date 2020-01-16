from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Input, Lambda
from DataPreparation import load_product_title_search_term_lists_words
import pandas as pd
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

window_size = 20

product_title_input, search_term_input = load_product_title_search_term_lists_words('train', window_size)
train_labels = pd.read_csv(r"C:\Users\micha\Desktop\train.csv", encoding="ISO-8859-1")['relevance']

x_product_train = product_title_input[:59000]
x_search_train = search_term_input[:59000]
x_product_val = product_title_input[59000:]
x_search_val = search_term_input[59000:]
y_train = train_labels[:59000]
y_val = train_labels[59000:]

model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))


#Input layers
input_product = Input(shape=(window_size, 1))
input_query = Input(shape=(window_size, 1))

encoded_term_product = model(input_product)
encoded_term_query = model(input_query)

distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
distance = distance_layer([encoded_term_product, encoded_term_query])

prediction = Dense(1, activation='sigmoid')(distance)

siamise_model = Model(inputs=[input_product, input_query], output=prediction)
model.compile(optimizer='adam', loss='mse')
siamise_model.compile(optimizer='adam', loss='mse')
print(siamise_model.summary())
mcp = ModelCheckpoint('model.h5',monitor='val_loss',verbose=1, save_best_only=True)
history = siamise_model.fit([x_product_train,x_search_train],y_train,validation_data=([x_product_val,x_search_val], y_val), batch_size=512, epochs=20, callbacks=[mcp])
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

