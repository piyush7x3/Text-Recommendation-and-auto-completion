from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
from fast_autocomplete import AutoComplete
from sympy import content
from .auto import autocomplete
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq
model = load_model('keras_next_word_model.h2')
WORD_LENGTH =2

def prepare_input(text,unique_words,unique_word_index):
    x = np.zeros((1, WORD_LENGTH, len(unique_words)))
    for t, word in enumerate(text.split()):
        x[0, t, unique_word_index[word]] = 2
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completions(text,unique_words,unique_word_index, n=3):
    if text == "":
        return("0")
    x = prepare_input(text,unique_words,unique_word_index)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] for idx in next_indices]


def index(request):
    
    ans = ['hello']
     
    if request.GET.get('word'):
        
        name = request.GET.get('word')

        if name[-1] == " ":
            file1 = open("sherlcok.txt","r+",errors = "ignore") 
            text = file1.read().lower()

            tokenizer = RegexpTokenizer(r'\w+')
            words = tokenizer.tokenize(text)

            unique_words = np.unique(words)
            unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

            
            prepare_input("It is".lower(),unique_words,unique_word_index)

            

            seq = " ".join(tokenizer.tokenize(name.lower())[0:6])
            
            ans = predict_completions(seq, unique_words,unique_word_index,3)
            

            






        else:
            

            ans = autocomplete.search(word=name, max_cost=3, size=3)


    return render(request, 'index.html',{
        'ans': ans,
    })
 


    

