from collections import Counter
import numpy as np
import re, json, os
from config import RegularConfig
from embeddings import Embeddings

class Tokenizer:
    def __init__(self, corpus):
        self.corpus = corpus
        self.config = RegularConfig()
        self.protected_tokens = self.corpus.protected_tokens
        self.vocabulary_path = os.path.join(self.config.PROJECT_ROOT, "data", "vocabulary.npz")
        self.embed = Embeddings()

        if os.path.exists(self.vocabulary_path):
            with np.load(self.vocabulary_path) as vocabulary:
                self.vocab = vocabulary["vocab"]
                self.rules = vocabulary["rules"]
                self.id_to_token = vocabulary["id_to_token"]
                self.token_to_id = vocabulary["token_to_id"]
            pass
    
    def words(self, text):
        newline = r"\n"        
        punctuation = r"([.,;:!?\"])"
        words = re.sub(newline, " ", text)
        words = re.sub(punctuation, r" \1 ", words)
        words = words.split()
        return [w for w in words if w not in self.protected_tokens]

    def characterize(self, text):
        chars = []
        for word in self.words(text):
            for char in word:
                chars.append(char)
        return chars

    def bpe_train(self):
        characters = self.characterize(self.corpus)
        self.vocab = sorted(list(set(characters)))
        self.vocab.append("<BEGIN>")
        self.vocab.append("<END>")
        self.vocab.append("<PAD>")
        self.vocab.append("<UNK>")
        
        self.rules = []

        while len(self.vocab) < self.config.vocab_length:
            
            a = 0
            pairs = []

            while a < len(characters) - 1:
                if not "_" in characters[a+1]:
                    pair = (characters[a], characters[a+1])
                    pairs.append(pair)
                a += 1

        
            rules = Counter(pairs).most_common()
            
            #rule selection

            rule = None

            for i in rules:
                rulestr = ""
                for char in i[0]: rulestr += char

                if rulestr in self.vocab:
                    continue
                else:
                    rule = i
                    self.vocab.append(rulestr)
                    break

            if rule is None:
                break

            i = 0
            characters2 = []
            while i < len(characters):
                if rule and i < len(characters) - 1 and characters[i] == rule[0][0] and characters[i+1] == rule[0][1]:
                    
                    merged = rule[0][0] + rule[0][1]
                    characters2.append(merged)
                    i += 2
                else:    
                    characters2.append(characters[i])
                    i += 1


            characters = characters2

            if rule: self.rules.append((rule[0])) 
            np.savez(self.vocabulary_path, vocab=self.vocab, rules=self.rules, id_to_token=self.id_to_token, token_to_id=self.token_to_id)

    def tokenize(self, input_str):
        tokens = self.characterize(input_str)

        #I will allow this to throw an exception manually, meaning it is out of sequence
        
        for rule in self.rules:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == rule[0] and tokens[i+1] == rule[1]:
                    merged = tokens[i] + tokens[i+1]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
        return tokens

    def encode(self, input_str):
        tokens = self.tokenize(input_str)
        
        encoded = []

        for token in tokens:
            try:
                encoded.append(self.token_to_id[token])
            except Exception:
                encoded.append(self.token_to_id["<UNK>"])
                
        return encoded

    def decode(self, input_nums):
        return [self.id_to_token[str(i)] for i in input_nums]

    def tpw_ratio(self):
        words = self.words(self.corpus)
        tokens = self.tokenize(self.corpus)

        if len(words) == 0:
            return 0
        
        tokens = [t for t in tokens if t != "_"]

        return len(tokens) / len(words)
        
    def tokenize_and_chunk_corpus(self):
        chunked = []
        encoded = self.encode(self.corpus)

        encoded = [self.token_to_id["<BEGIN>"]] + encoded

        remainder = len(encoded) % self.config.max_seq_length
        for i in range(0, len(encoded), self.config.max_seq_length):
            chunk = [0] + encoded[i:i+self.config.max_seq_length-1]
            chunked.append(chunk)
        if remainder != 0:
            chunked.append(encoded[len(encoded)-self.config.max_seq_length : len(encoded)])
        return chunked
    
    def save_embeddings(self):
        np.save(self.embeddings_path, self.E)