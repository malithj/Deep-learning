import torch
import unicodedata
import re
import pandas as pd
import pickle
import os
from torch.utils.data import Dataset, DataLoader

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
    
    
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def addSentence(self, sentence):
        sentence = self.normalizeString(sentence)
        for word in sentence.split(' '):
            word_ = word.strip('\n')
            word_ = word_.strip('\t')
            self.addWord(word_)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def indexesFromSentence(self, sentence):
        return [self.word2index[word] if word in self.word2index else 0 for word in sentence.split(' ')]

    def tensorFromSentence(self, sentence):
        sentence = self.normalizeString(sentence)
        indexes = self.indexesFromSentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

    def sentenceFromTensor(self, tensor):
        list_ = list()
        for idx in tensor:
            if idx in self.index2word:
                list_.append(str(self.index2word[idx]))
            else:
                list_.append(str(self.index2word[0]))
        return list_

    def __get_state__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)


class TextMappingDataset(Dataset):
    def __init__(self, english_file, foriegn_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.english_txt = pd.read_csv(english_file, delimiter='\n')
        self.foriegn_txt = pd.read_csv(foriegn_file, delimiter='\n')
        self.input_lang = Lang("English")
        self.output_lang = Lang("Foriegn")
        self.root_dir = root_dir
        self.transform = transform
        self.__language_init__(english_file, foriegn_file)

    def __language_init__(self, english_file, foriegn_file):
        print("Processing Data")
        file_name_e = english_file.split('/')[2].split('.')[0] + "_" + english_file.split('/')[2].split('.')[1]
        file_name_f = foriegn_file.split('/')[2].split('.')[0] + "_" + foriegn_file.split('/')[2].split('.')[1]

        if os.path.exists('./model/{0:}.p'.format(file_name_e)):
            self.input_lang = pickle.load(open('./model/{0:}.p'.format(file_name_e), "rb"))
            self.output_lang = pickle.load(open('./model/{0:}.p'.format(file_name_f), "rb"))
            print("Loaded vocabulary from pickled sources")
            return       

        for i in range(len(self.english_txt)):
            self.input_lang.addSentence(str(self.english_txt.iloc[i]))
            self.output_lang.addSentence(str(self.foriegn_txt.iloc[i]))
        print("ENC_VOCAB: {0:}".format(self.input_lang.n_words))
        print("DEC_VOCAB: {0:}".format(self.output_lang.n_words))
        print("Loaded vocabulary from processing data")
        pickle.dump(self.input_lang, open("./model/{0:}.p".format(file_name_e), "wb"))
        pickle.dump(self.output_lang, open("./model/{0:}.p".format(file_name_f), "wb"))

    def __len__(self):
        return len(self.english_txt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _english_idx = self.input_lang.tensorFromSentence(self.english_txt.iloc[idx].to_string())
        _foriegn_idx = self.output_lang.tensorFromSentence(self.foriegn_txt.iloc[idx].to_string())
        sample = {'english_txt': _english_idx, 'foriegn_txt': _foriegn_idx}

        if self.transform:
            sample = self.transform(sample)

        return sample
