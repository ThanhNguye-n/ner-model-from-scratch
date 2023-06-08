import numpy as np
import tensorflow as tf
from nltk import word_tokenize
from tensorflow.keras.utils import pad_sequences
from utils_function import getCasing, createTensor, addCharInformation

class NER():

    def __init__(self, model_name):
        self.char2Idx = np.load("char2Idx.npy", allow_pickle=True).item()
        self.case2Idx = np.load("case2Idx.npy", allow_pickle=True).item()
        self.word2Idx = np.load("word2Idx.npy", allow_pickle=True).item()
        self.idx2Label = np.load("idx2Label.npy", allow_pickle=True).item()
        self.model = tf.keras.models.load_model(f'{model_name}.h5')

    def padding(self,Sentence):
        Sentence[2] = pad_sequences(Sentence[2], 40 ,padding='post')
        return Sentence

    def predict(self,Sentence):
        words =  word_tokenize(Sentence)
        Sentence = word_tokenize(Sentence)
        Sentence = addCharInformation(Sentence)
        Sentence = self.padding(createTensor(Sentence,self.word2Idx,self.case2Idx,self.char2Idx))
        tokens, casing,char = Sentence
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = self.model.predict([tokens, casing,char], verbose=False)[0]
        pred = pred.argmax(axis=-1)
        pred = [self.idx2Label[x].strip() for x in pred]
        return list(zip(words,pred))