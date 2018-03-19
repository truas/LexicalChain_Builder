'''
Created on Mar 16, 2018

@author: Terry Ruas
'''
class SemanticSynsetData(object):
    def __init__(self):
        self.synset_reations = dict()

class ChainData(object):
    def __init__(self):
        self.chain_id = None
        self.chain_token = [] #List of TokenData of the Chain
        self.chain_position = None
        self.chain_relations = SemanticSynsetData() #all the synsets in that chain

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