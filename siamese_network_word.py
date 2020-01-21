from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Input, Lambda
from DataPreparation import load_product_title_search_term_lists_words
import pandas as pd
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import gensim
import pickle as pkl
from gensim.models import Word2Vec


def to_embbeded(list_of_ngrams, w2v_model):
    all_new_val = []
    for val in list_of_ngrams:
        new_val = []
        for gram in val:
            new_val.append(w2v_model.wv.get_vector(gram))
        all_new_val.append(new_val)

    return np.array(all_new_val)


window_size = 20

product_title_input, search_term_input = load_product_title_search_term_lists_words('train', 20)
train_labels = pd.read_csv(r"C:\Users\odedblu\Desktop\train.csv", encoding="ISO-8859-1")['relevance']

x_product_train = product_title_input[:59000]
x_search_train = search_term_input[:59000]
x_product_val = product_title_input[59000:]
x_search_val = search_term_input[59000:]
y_train = train_labels[:59000]
y_val = train_labels[59000:]

combaind_list = x_product_train + x_search_train + x_product_val + x_search_val
gen_model = Word2Vec(combaind_list, size=30, window=5, min_count=0, workers=10)
gen_model.train(combaind_list, total_examples=len(combaind_list), epochs=10)

pkl.dump(gen_model, open('gen_model.pkl','wb'))


x_product_train_embbeded = to_embbeded(x_product_train, gen_model)
x_search_train_embbeded = to_embbeded(x_search_train, gen_model)
x_product_val_embbeded = to_embbeded(x_product_val, gen_model)
x_search_val_embbeded = to_embbeded(x_search_val, gen_model)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(window_size, 30)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))

# Input layers
input_product = Input(shape=(window_size, 30))
input_query = Input(shape=(window_size, 30))

encoded_term_product = model(input_product)
encoded_term_query = model(input_query)

distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
distance = distance_layer([encoded_term_product, encoded_term_query])

prediction = Dense(1, activation='sigmoid')(distance)

siamise_model = Model(inputs=[input_product, input_query], output=prediction)
model.compile(optimizer='adam', loss='mse')
siamise_model.compile(optimizer='adam', loss='mse')
print(siamise_model.summary())
mcp = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)
history = siamise_model.fit([x_product_train_embbeded, x_search_train_embbeded], y_train,
                            validation_data=([x_product_val_embbeded, x_search_val_embbeded], y_val), batch_size=128, epochs=50,
                            callbacks=[mcp])
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
