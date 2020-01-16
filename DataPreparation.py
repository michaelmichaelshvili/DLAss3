# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer

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
# dict = load_char_dictionary()


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


def load_product_title_search_term_lists(train_or_test, window_size):
    product_title_list = pkl.load(open(f'{train_or_test}_product_title_list_window{window_size}.pkl', 'rb'))
    search_term_list = pkl.load(open(f'search_term_list_window{window_size}.pkl', 'rb'))
    return product_title_list, search_term_list

def prepere_X_data():
    train_df = pd.read_csv(os.path.join(path_dir, 'train.csv'), encoding="ISO-8859-1")
    # test_df = pd.read_csv(os.path.join(path_dir, 'test.csv'), encoding="ISO-8859-1")
    corpus = []
    for idx,row in train_df.iterrows():
        product_title = row['product_title']
        search_term = row['search_term']
        corpus.append(str(product_title))
        corpus.append(str(search_term))

    count_vectoraizer = CountVectorizer(analyzer='char', encoding="ISO-8859-1")
    vectorizer_output = count_vectoraizer.fit_transform(corpus)

    vectorizer_output = vectorizer_output.toarray()
    vectorizer_output_arr = []
    for i in range (0,len(vectorizer_output),2):
        vectorizer_output_arr.append(np.concatenate((vectorizer_output[i],vectorizer_output[i+1]),axis=None))
    return vectorizer_output_arr

def prepere_X_data_ngram3():
    train_df = pd.read_csv(os.path.join(path_dir, 'train.csv'), encoding="ISO-8859-1")
    # test_df = pd.read_csv(os.path.join(path_dir, 'test.csv'), encoding="ISO-8859-1")
    corpus = []
    for idx,row in train_df.iterrows():
        product_title = row['product_title']
        search_term = row['search_term']
        corpus.append(str(product_title))
        corpus.append(str(search_term))

    count_vectoraizer = CountVectorizer(analyzer='char', encoding="ISO-8859-1", ngram_range=(3, 3))
    count_vectoraizer.fit_transform(corpus)
    grams = count_vectoraizer.get_feature_names()
    dictionary = {}
    for idx, key in enumerate(grams):
        dictionary[key] = idx + 1
    pkl.dump(dictionary, open('dictionary_3_gram.pkl', 'wb'))



def str_to_array_of_ngrams(string, word_dictionary, window_size):
    string = ' '.join(string.split())
    string = list(string.lower())
    res = []
    for i in range(0,len(string)-2):
        gram = word_dictionary[''.join(string[i:i+3])]
        res.append(gram)
    # string = string.reshape((len(string), 1))
    if window_size < len(res):
        res = res[:window_size]
    elif window_size == len(res):
        res = res
        pass
    else:
        zevel = [0] * (window_size - len(word_dictionary))
        zevel.extend(res)
        res = zevel
    res = np.array(res).reshape((len(res), 1))
    return res

def prepare_network_input_X_ngram(train_or_test, window_size):
    word_dictionary = pkl.load(open('dictionary_3_gram.pkl', 'rb'))
    if train_or_test is 'train':
        df = pd.read_csv(os.path.join(path_dir, 'train.csv'), encoding="ISO-8859-1")
    else:
        df = pd.read_csv(os.path.join(path_dir, 'test.csv'), encoding="ISO-8859-1")

    product_title_list = []
    search_term_list = []

    for idx, row in df.iterrows():
        product_title = row['product_title']
        search_term = row['search_term']
        product_title_array = str_to_array_of_ngrams(product_title, word_dictionary, window_size)
        product_title_list.append(product_title_array)
        search_term_array = str_to_array_of_ngrams(search_term, word_dictionary, window_size)
        search_term_list.append(search_term_array)
    pkl.dump(product_title_list, open(f'{train_or_test}_product_title_list_word_window{window_size}.pkl', 'wb'))
    pkl.dump(search_term_list, open(f'{train_or_test}_search_term_list_word_window{window_size}.pkl', 'wb'))

def load_product_title_search_term_lists_words(train_or_test, window_size):
    product_title_list = pkl.load(open(f'{train_or_test}_product_title_list_word_window{window_size}.pkl', 'rb'))
    search_term_list = pkl.load(open(f'{train_or_test}search_term_list_word_window{window_size}.pkl', 'rb'))
    return product_title_list, search_term_list