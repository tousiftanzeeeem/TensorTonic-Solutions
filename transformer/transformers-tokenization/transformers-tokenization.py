import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 4
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.word_to_id[self.pad_token] = 0
        self.id_to_word[0] = self.pad_token
        self.word_to_id[self.unk_token] = 1
        self.id_to_word[0] = self.unk_token
        self.word_to_id[self.bos_token] = 2
        self.id_to_word[0] = self.bos_token
        self.word_to_id[self.eos_token] = 3
        self.id_to_word[0] = self.pad_token

    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        vocabs = []
        for str in texts:
            words = str.lower().split(' ')
            vocabs.extend(words)
        vocabs = sorted(list(set(vocabs)))
        id = 4
        for i in vocabs:
            self.word_to_id[i] = id
            self.id_to_word[id] = i
            id = id + 1

        self.vocab_size = self.vocab_size + len(vocabs) 
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = []
        if text == '':
            return []
        for i in text.lower().split(' '):
            tokenid = self.word_to_id.get(i,1)
            tokens.append(tokenid)
        return tokens
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        out = []
        for i in ids:
            out.append(self.id_to_word.get(i,"<UNK>"))
        return ' '.join(x for x in out)
            
            
        
