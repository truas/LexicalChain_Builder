'''
Created on Mar 16, 2018

@author: Terry Ruas
'''
class SemanticSynsetData(object):
    def __init__(self):
        self.synset_relations = dict()

class ChainData(object):
    def __init__(self):
        self.chain_id = idData()
        self.prospective_tokens = [] #List of idData of the Chain
        self.chain_relation_tokens = SemanticSynsetData() #all the synset-related(SSR) in this chain

#same structure as TokenData, but no hard-constructor, easier to differentiate an ID token
class idData(object):
    def __init__(self):
        self.iword = None
        self.isyn=None
        self.ioffset=None
        self.ipos=None

class TokenData(object):#used to read information from -synset-file into object
    def __init__(self, word, syn, offset, pos):
        self.word=word
        self.syn=syn
        self.offset=offset
        self.pos=pos
        
class DocumentData(object):
    def __init__(self):
        self.tokens = [] #list of TokenData of the document
        self.chains = [] #list of ChainData of the document