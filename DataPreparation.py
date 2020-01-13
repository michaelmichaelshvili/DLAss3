# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import pickle as pkl

path_dir = r'C:\Users\micha\Desktop'


def make_cahe_dictionaty():
    train_df = pd.read_csv(os.path.join(path_dir, 'train.csv'), encoding="ISO-8859-1")
    test_df = pd.read_csv(os.path.join(path_dir, 'test.csv'), encoding="ISO-8859-1")
    unique_chars = set()
    for idx, row in train_df.iterrows():
        text = row['product_title'] + row['search_term']
        uniqe_list = set(text.lower())
        unique_chars = unique_chars.union(uniqe_list)

    for idx, row in test_df.iterrows():
        text = row['product_title'] + row['search_term']
        uniqe_list = set(text.lower())
        unique_chars = unique_chars.union(uniqe_list)

    dictionary_encoding = {}
    for idx, val in enumerate(unique_chars):
        dictionary_encoding[val] = idx
    pkl.dump(dictionary_encoding, open('char_to_code.pkl', 'wb'))


def load_char_dictionary():
    return pkl.load(open('char_to_code.pkl', 'rb'))


# make_cahe_dictionaty()
dict = load_char_dictionary()


def str_to_array_of_array(string, char_dictionary, window_size):
    string = list(string.lower())
    string = list(map(lambda x: char_dictionary[x], string))
    # string = string.reshape((len(string), 1))
    if window_size < len(string):
        string = string[:window_size]
    elif window_size == len(string):
        string = string
        pass
    else:
        zevel = [len(char_dictionary)]*(window_size - len(string))
        zevel.extend(string)
        string = zevel
    res = np.array(string).reshape((len(string), 1))
    return res


def prepare_network_input_X(train_or_test, char_dictionary, window_size):
    if train_or_test is 'train':
        df = pd.read_csv(os.path.join(path_dir, 'train.csv'), encoding="ISO-8859-1")
    else:
        df = pd.read_csv(os.path.join(path_dir, 'test.csv'), encoding="ISO-8859-1")

    product_title_list = []
    search_term_list = []

    for idx, row in df.iterrows():
        product_title = row['product_title']
        search_term = row['search_term']
        product_title_array = str_to_array_of_array(product_title, char_dictionary, window_size)
        product_title_list.append(product_title_array)
        search_term_array = str_to_array_of_array(search_term, char_dictionary, window_size)
        search_term_list.append(search_term_array)
    pkl.dump(product_title_list, open(f'{train_or_test}_product_title_list_window{window_size}.pkl', 'wb'))
    pkl.dump(search_term_list, open(f'search_term_list_window{window_size}.pkl', 'wb'))


def load_product_title_search_term_lists(window_size):
    product_title_list = pkl.load(open(f'product_title_list_window{window_size}.pkl', 'rb'))
    search_term_list = pkl.load(open(f'search_term_list_window{window_size}.pkl', 'rb'))
    return product_title_list, search_term_list

