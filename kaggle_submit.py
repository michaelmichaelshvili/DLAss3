from keras.models import load_model
import numpy as np
import pandas as pd
import DataPreparation
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle as pkl


def to_embbeded(list_of_ngrams):
    w2v_model = pkl.load(open('gen_model.pkl', 'rb'))
    all_new_val = []
    for val in list_of_ngrams:
        new_val = []
        for gram in val:
            try:
                new_val.append(w2v_model.wv.get_vector(gram))
            except:
                new_val.append([0] * 30)
        all_new_val.append(new_val)

    return np.array(all_new_val)


def char_level():
    model_path = r'char_lvl_model.h5'
    model_for_prediction = load_model(model_path)
    product_title_input_test, search_term_input_test = DataPreparation.load_product_title_search_term_lists('test', 20)
    product_title_input_train, search_term_input_train = DataPreparation.load_product_title_search_term_lists('train',
                                                                                                              20)
    return product_title_input_train, search_term_input_train, product_title_input_test, search_term_input_test, model_for_prediction


def word_level():
    model_path = r'model.h5'

    model_for_prediction = load_model(model_path)
    product_title_input_test, search_term_input_test = DataPreparation.load_product_title_search_term_lists_words(
        'test', 20)
    product_title_input_train, search_term_input_train = DataPreparation.load_product_title_search_term_lists_words(
        'train', 20)
    return product_title_input_train, search_term_input_train, product_title_input_test, search_term_input_test, model_for_prediction


product_title_input_train, search_term_input_train, product_title_input_test, search_term_input_test, model = word_level()
train_labels = pd.read_csv(r"C:\Users\odedblu\Desktop\train.csv", encoding="ISO-8859-1")['relevance']
x_product_train = product_title_input_train[:59000]
x_search_train = search_term_input_train[:59000]
x_product_val = product_title_input_train[59000:]
x_search_val = search_term_input_train[59000:]
x_product_test = product_title_input_test
x_search_test = search_term_input_test
y_train = train_labels[:59000]
y_val = train_labels[59000:]
y_test = pd.read_csv(r"C:\Users\odedblu\Desktop\solution.csv")['relevance']
index_minus = []
for idx, i in enumerate(y_test):
    if i == -1:
        index_minus.append(idx)
x_product_test = [i for idx, i in  enumerate(x_product_test) if idx not in index_minus]
x_search_test = [i for idx, i in  enumerate(x_search_test) if idx not in index_minus]
y_test = [i for i in y_test if i!=-1]

x_product_train_embbeded = to_embbeded(x_product_train)
x_search_train_embbeded = to_embbeded(x_search_train)
x_product_val_embbeded = to_embbeded(x_product_val)
x_search_val_embbeded = to_embbeded(x_search_val)
x_search_test_embbeded = to_embbeded(x_search_test)
x_search_test_embbeded = to_embbeded(x_search_test)

train_pred = model.predict([x_product_train_embbeded, x_search_train_embbeded])
val_pred = model.predict([x_product_val_embbeded, x_search_val_embbeded])
test_pred = model.predict([x_search_test_embbeded, x_search_test_embbeded])

train_mse, train_mae = mean_squared_error(y_train, train_pred), mean_absolute_error(y_train, train_pred)
val_mse, val_mae = mean_squared_error(y_val, val_pred), mean_absolute_error(y_val, val_pred)
test_mse, test_mae = mean_squared_error(y_test, test_pred), mean_absolute_error(y_test, test_pred)

print('train:')
print('mse=' + str(train_mse))
print('mae=' + str(train_mae))
print('val:')
print('mse=' + str(val_mse))
print('mae=' + str(val_mae))
print('test:')
print('mse=' + str(test_mse))
print('mae=' + str(test_mae))
