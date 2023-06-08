# From: https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs

from tensorflow.keras.utils import pad_sequences
import numpy as np
import random

def getCasing(word, caseLookup):
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup[casing]


def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len


def createMatrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []
        flag = False

        for word, char, label in sentence:
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            charIdx = []

            ADD_TO_DATA = True
            for x in char:
                if x in char2Idx.keys():
                    charIdx.append(char2Idx[x])
                else:
                    flag = True
                    ADD_TO_DATA = False
                    break
            if flag:
                break
            # Get the label and map to int
            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, case2Idx))
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])

        if ADD_TO_DATA:
            dataset.append([wordIndices, caseIndices, charIndices, labelIndices])

    return dataset



# 0-pads all words
def padding(Sentences):
    maxlen = 40
    for sentence in Sentences:
        char = sentence[2]
        for x in char:
            maxlen = max(maxlen, len(x))
    for i, sentence in enumerate(Sentences):
        Sentences[i][2] = pad_sequences(Sentences[i][2], 40, padding='post')
    return Sentences


def iterate_minibatches(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        caseing = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, c, ch, l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            caseing.append(c)
            char.append(ch)
            labels.append(l)

        yield np.asarray(labels), np.asarray(tokens), np.asarray(caseing), np.asarray(char)


def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len

def createTensor(sentence, word2Idx, case2Idx, char2Idx):
        unknownIdx = word2Idx['UNKNOWN_TOKEN']

        wordIndices = []
        caseIndices = []
        charIndices = []

        for word, char in sentence:
            word = str(word)
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
            charIdx = []
            for x in char:
                if x in char2Idx.keys():
                    charIdx.append(char2Idx[x])
                else:
                    charIdx.append(char2Idx['UNKNOWN'])
            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, case2Idx))
            charIndices.append(charIdx)

        return [wordIndices, caseIndices, charIndices]

def addCharInformation(sentence):
        return [[word, list(str(word))] for word in sentence]