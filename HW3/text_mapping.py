import torch
import unicodedata
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader

SOS_token = 0
EOS_token = 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
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
        return [self.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, sentence):
        sentence = self.normalizeString(sentence)
        indexes = self.indexesFromSentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

    def sentenceFromTensor(self, tensor):
        list_ = list()
        for idx in tensor:
            if idx in self.index2word:
                list_.append(self.index2word[idx])
        print(list_)
        return list_

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters

    def normalizeString(self, s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s


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
        self.__language_init__()

    def __language_init__(self):
        print("Processing Data")
        for i in range(len(self.english_txt)):
            self.input_lang.addSentence(str(self.english_txt.iloc[i]))
            self.output_lang.addSentence(str(self.foriegn_txt.iloc[i]))
        print("ENC_VOCAB: {0:}".format(self.input_lang.n_words))
        print("DEC_VOCAB: {0:}".format(self.output_lang.n_words))

    def __len__(self):
        return len(self.english_txt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _english_idx = self.input_lang.tensorFromSentence(str(self.english_txt.iloc[idx]))
        _foriegn_idx = self.output_lang.tensorFromSentence(str(self.foriegn_txt.iloc[idx]))
        sample = {'english_txt': _english_idx, 'foriegn_txt': _foriegn_idx}

        if self.transform:
            sample = self.transform(sample)

        return sample
