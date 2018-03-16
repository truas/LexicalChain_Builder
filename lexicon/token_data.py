'''
Created on Mar 16, 2018

@author: Terry Ruas
'''
class ChainData(object):
    def __init__(self):
        self.chain_id = None
        self.chain_token = None
        self.chain_position = None

class TokenData(object):
    def __init__(self, word, syn, offset, pos):
        self.word=word
        self.syn=syn
        self.offset=offset
        self.pos=pos
        
class DocumentData(object):
    def __init__(self):
        self.tokens = [] #list of TokenData of the document
        self.chains = [] #list of ChainData of the document