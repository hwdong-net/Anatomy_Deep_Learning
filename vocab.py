import numpy as np
from collections import defaultdict
SOS_token = 0
EOS_token = 1
UNK_token = 2
    
class Vocab:
    def __init__(self,min_count=1,corpus = None): 
        self.min_count = 1
        self.word2count = {}
        self.word2index = {"SOS":0,"EOS":1, "UNK":2}        
        self.index2word = {0: "SOS", 1: "EOS",2: "UNK"}
        self.n_words = 3  # Count SOS and EOS
        if corpus is not None:
            for sentence in corpus:
                self.addSentence(sentence)
            self.build()               

    def addSentence(self, sentence):
        if isinstance(sentence,str):
            for word in sentence.split(' '):
                self.addWord(word)                
        else:
            for word in sentence:
                self.addWord(word)              

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
            
    def build(self):
        for word in self.word2count:
            if self.word2count[word]<self.min_count:
                self.word2index[word] = UNK_token
            else:
                self.word2index[word] = self.n_words 
                self.index2word[self.n_words] = word
                self.n_words += 1
                

#----------------readpairs----------------------- 
import unicodedata
import re
import random               
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

def readLangs(lang2lang_file, reverse=False):
    print("Reading lines...")
    
    lines = open(lang2lang_file, encoding='utf-8').\
        read().strip().split('\n')
    
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')][:2] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]  
      
    return pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def read_pairs(lang2lang_file, reverse=False):
    pairs = readLangs(lang2lang_file,reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    return pairs