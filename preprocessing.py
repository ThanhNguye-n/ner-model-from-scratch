import pandas as pd
import re

class Processing_DataFrame(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def remove_label(self, labels=None):
        """Remove label which only has a few samples. DEFAULT='Tag' """
        for label in labels:
            for i in self['Intent']:
                if label in i:
                    tag = self['Intent'] == i
                    self.drop(self[tag].index, inplace=True)

        self.reset_index(drop=True, inplace=True)

    def remove_higher_len(self, len_threshold=40):
        """Remove sentence which has a length of words higher than len_threshold """
        for i in self['Question']:
            if len(i.split()) > len_threshold:
                filter = self['Question'] == i
                self.drop(self[filter].index, inplace=True)

        self.reset_index(drop=True, inplace=True)

    def preprocessing_ner(self):
        """Create column Preprocess_Question by taking index, label from Parameters and replace them into Question
        Ex:  Senator McCain could secure the Republican Party nomination with victories Tuesday
        ---> B-per   I-per  could secure the B-org      I-org nomination with victories B-tim
        """

        list_process_ques = []
        list_labels = []

        for i in range(self.shape[0]):
            para = self['Parameters'][i]
            lists = re.findall(r'\[[^]]*', para) # find a sequence like [0, 7, "B-per"]  ------  \[ + [^]]* = r'\[(\d+),\s(\d+),\s"([A-Za-z-]+)"]'
            list_index = [x.replace('[','').split(',') for x in lists]
            try:
                list_index = sorted(list_index, key=lambda x: int(x[0])) # sort by index first index
            except:
                pass

            sentence = self['Question'][i]
            new_sentence = ""
            current_index = 0
            for i in list_index:
                try:
                    start = int(i[0])
                    end = int(i[1])
                    label = str(i[2].replace('"','').strip())
                    list_labels.append(label)
                    new_sentence += sentence[current_index:start] + label
                    current_index = end
                except:
                    pass
            new_sentence += sentence[current_index:]
            list_process_ques.append(new_sentence)

        self['Preprocess_Question'] = list_process_ques
        self['Question'] = self['Question'].apply(lambda x: x.strip())

        return set(list_labels)

    def create_target(self, list_labels):
        """Create target columns, if a word in label_list, it would be retained. Otherwise, it would tag as label O"""

        list_Y = []
        for sent in self['Preprocess_Question']:
            pre_sent = [x if x in list_labels else 'O' for x in sent.strip().split(' ')]
            combine = " ".join(pre_sent)
            list_Y.append(combine)

        self['Target'] = list_Y

    def split_word_char_label(self):
        """Split a sentence into a list contains a word, charater and label. Return in 'input' column """

        sentence = []
        for idx, sent in enumerate(self['Question']):
            temp = []
            for word in sent.split(" "):
                temp.append([word])
            sentence.append(temp)

        temp_set = sentence.copy()

        for i, sentence in enumerate(sentence):

            label = self['Target'][i].split(" ")

            for j, word in enumerate(sentence):
                temp_set[i][j].append([c for c in word[0]])
                temp_set[i][j].append(label[j])

        self['input'] = temp_set