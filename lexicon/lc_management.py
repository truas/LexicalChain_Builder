#!/usr/bin/python3
'''
Created on Mar 19, 2018

@author: Terry Ruas

'''
#imports
import numpy
import sys
import random
import numpy as np
from scipy import spatial
from nltk.corpus import wordnet as wn

#local-import
from lexicon import token_data as td



#Global - Definitions
PRECISION_COS = 7 #precision for cosine-distance/similairty
CHUNK_SIZE = 4 #size of chain blocks in fixed lexical chains
LOW_CAP = -0.5 #lower bound of normal distribution value
HIGH_CAP = 0.5 #upper bound of normal distribution value


#definitions

#synset node relationships in WordNet
NYMS = ['hypernyms','instance_hypernyms' , 'hyponyms', 'instance_hyponyms', 
        'member_holonyms', 'substance_holonyms', 'part_holonyms', 
        'member_meronyms', 'substance_meronyms', 'part_meronyms',
        'attributes','entailments','causes', 'also_sees', 'verb_groups', 'similar_tos',
        'topic_domains', 'region_domains', 'usage_domains']

#Part od speech weights
POS_W = {'n':1.0, 'v':1.0, 'r':1.0, 'a':1.0, 's':1.0}

'''
#===============================================================================
# FlexChain Block - START
#===============================================================================
'''

def build_FlexChain(data_tokens, vec_model):
    if not (data_tokens): return (False) # in case a empty document is provided we will skip it
    
    #initialize chain
    flex_chains = []
    flex_chains.append(start_FlexChain(data_tokens[0])) #the first element is already in the chain    

    for counter, token in enumerate(data_tokens,1):#visit every token except the first   
        adopt = False
        #last element of the chain
        last = len(flex_chains)-1
        tmp_iddata = td.idData()
        #tmp chains building block
        tmp_iddata.iword = token.iword 
        tmp_iddata.isyn = wn.synset_from_pos_and_offset(token.ipos, token.ioffset)  # @UndefinedVariable
        tmp_iddata.ioffset = token.ioffset        
        tmp_iddata.ipos = token.ipos
        tmp_ssr = build_synset_relations(tmp_iddata.ioffset, tmp_iddata.ipos) # retrieve related synsets from synset in the text
        
        #if there was a match between both SSR we will add this synset to the chain 
        for key in set(tmp_ssr) & set(flex_chains[last].chain_relation_tokens):
            if tmp_ssr[key] == flex_chains[last].chain_relation_tokens[key]:
                adopt = True
                break 
        
        if adopt:#incorporate synset into current chain
            
            flex_chains[last].chain_relation_tokens = relatedSynsetChainUpdate(tmp_ssr,flex_chains[last].chain_relation_tokens)
            
            flex_chains[last].prospective_tokens.append(tmp_iddata)
        else:#calculate current chain representative and create a new one to start a new chain block
            flex_chains[last].chain_id = representProspectiveChain(flex_chains[last], vec_model)
            new_chain =  start_FlexChain(token)
            flex_chains.append(new_chain) 

    return(flex_chains)
#build flexible lexical chain

def start_FlexChain(data_token):
    chain_start = td.ChainData()
    chain_iddata = td.idData()
    #chain possible representative
    chain_iddata.iword = data_token.iword
    chain_iddata.isyn = wn.synset_from_pos_and_offset(data_token.ipos, data_token.ioffset)  # @UndefinedVariable
    chain_iddata.ioffset = data_token.ioffset
    chain_iddata.ipos = data_token.ipos
    chain_start.prospective_tokens.append(chain_iddata)
    #chains related synsets
    chain_start.chain_relation_tokens = build_synset_relations(chain_iddata.ioffset, chain_iddata.ipos)
    #chain id  setup
    chain_start.chain_id = chain_iddata
    
    return(chain_start)
#initializes first element of a chain  and set this chains id with the first token (iword#ioffset#ipos)

def build_synset_relations(offset, pos):
    relation_synsets = dict()
    synset = wn.synset_from_pos_and_offset(pos, offset) # @UndefinedVariable
    relation_synsets[synset] = 1 #the synset itself is part of the related ones
        
    for nym in NYMS:#for all synset-lexical-relations in wordnet
        try:
            tmp_items = getattr(synset, nym)()
            if not tmp_items: continue #if that relation retrieves no synsets just skip the whole thing
            for item in tmp_items:
                if item not in relation_synsets: #if this is a new synset include on the dictionary
                    relation_synsets[item] = 1
                else:
                    relation_synsets[item] += 1
        except:
            continue        
                
    return (relation_synsets)
#produces a list of synsets from all the *_NYMS from that synset - produces SSR from a synset(ioffset,ipos)

def relatedSynsetChainUpdate(current_related_synset, chain_related_synset):
    #Python >= 3.5 Compatible
    #MINOR_python_version = 5
    #update_related_synsets = {**current_related_synset, **chain_related_synset}
  
    #Python <= 3.4 Compatible
    update_related_synsets = current_related_synset.copy()
    update_related_synsets.update(chain_related_synset)
    
    return(update_related_synsets)      
#Merge dictionaries of related synsets. In python 3.5+ {**dict, **dict}, 
#in python 3.4 or less need to use dict.copy and dict.update    

'''
#===============================================================================
# FlexChain Block - END
#===============================================================================
'''
'''
#===============================================================================
#  FixedChain Block - Start
#===============================================================================
'''
def build_FixedChain(data_tokens, vec_model, chunk_size):
    if not (data_tokens): return (False)# in case a empty document is provided we will skip it
    
    chunk = checkChainSize(chunk_size) #validates if a proper chunk-size was provided
    fixed_chains = []#list of fixed chains
    tmp_chains = []
   
    tmp_chains = chunker(data_tokens, chunk)#slice of the data-tokens in document-synset
    chains_data = convertFixedChain(tmp_chains, vec_model) #returns a list of chainData - for FX we just care about the prospective_tokens
    
    #representing fixed chain
    for chunk in chains_data:
        fixed_chains.append(chunk)
        fixed_chains[-1].chain_id = representProspectiveChain(chunk, vec_model)
    return(fixed_chains)      
#build fixed lexical chains based on chunks


def convertFixedChain(tmp_chains, vec_model):
    fx_chains = []
    for tmp_chain in tmp_chains:
        fx_chain = td.ChainData()
        fx_chain.prospective_tokens = tmp_chain
        fx_chains.append(fx_chain)
    return (fx_chains)
#Transforms list of chunk-chains into a list of ChainData, with each chunk as a prospective_tokens
#prospective tokens are the objects used to calculate the representative items in that chain

def chunker(seq, size):
    if(len(seq) >= size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    else:
        return [seq]
#slices list accordingly to a given size, if list of data-tokens is smaller than chunk-size
#make a list of all data-tokens ([[data-tokens]])

def checkChainSize(chain_size):
    if(chain_size):
        return (chain_size)
    else:
        return (CHUNK_SIZE)
#validates chunk size 

'''
#===============================================================================
#  FixedChain Block - End
#===============================================================================
'''

'''
#===============================================================================
# CHAIN REPRESENTATION
#===============================================================================
'''
def representProspectiveChain(current_chain, vec_model):
    vecs, weight_pos = calculateChainRepresentative(current_chain, vec_model) 
    current_chain_avg = np.average(vecs, weights = weight_pos, axis=0) #weighted average of the current chain - weights are the values on POS_W
    chain_rep = closest_synset_rep(current_chain.prospective_tokens, current_chain_avg, vec_model) 
    return(chain_rep)
#given a chain with several synsets select one to represent it

def closest_synset_rep(prospective_ids, chain_avg, vec_model):
    highest_sofar = -float('inf')
    choice = 0
    for i, candidates in enumerate(prospective_ids):
        key = makeKey(candidates.iword,candidates.ioffset,candidates.ipos)
        cand,_ = retrieveModelKey(key, vec_model) #vector for given key in a nmodel  (flag not used here)       
        tmp = cosine_similarity(chain_avg, cand)
        #keep the index of the element with the highest cos-sim with the average in the chain
        if tmp > highest_sofar: #
            highest_sofar = tmp
            choice = i
            
    return(prospective_ids[choice]) #an idData will be returned to represent the current chain   
#elects the idData(synset-key) with the highest cosine against the average of current chain prospective synsets            
#We take the first element closest to the average in the prospective chain, even if
#there are others with the same value (this is an extremely rare situation given the dim)

def calculateChainRepresentative(current_chain, vec_model):
    vecs = [] #synset vectors
    weight_pos =[] #weight of synset vetors based on POS
    for item in current_chain.prospective_tokens:
            key = makeKey(item.iword,item.ioffset,item.ipos)
            vec,flag = retrieveModelKey(key, vec_model)
            wei = weightPOS(flag, item.ipos) #weight (ipos) based on the existence of a vector
            vecs.append(vec)
            weight_pos.append(wei)
    return(vecs,weight_pos)
#given a list of synsets it retrieves its vectors and respective weights
   
def makeKey(word, offset, pos):
    return(word+'#'+str(offset)+'#'+pos)
#build a key composed by iword#ioffset#POS (to be used in a embeddings model)
#it doesnt matter if this comes from token data or idData

def retrieveModelKey(key,vec_model):
    try:
        vec = vec_model.word_vec(key)
        flag = True
    except KeyError:
        vec = np.random.uniform(low=LOW_CAP, high=HIGH_CAP, size = vec_model.vector_size)
        flag = False
    return(vec,flag)
#retrieves a vector of x-dimensions for a given key in the trained model and TRUE
#else, returns a pseudo-random normal distribution between (LOW_CAP,HIGH_CAP) and FALSE     

def weightPOS(vec_found, ipos):
    if(vec_found):
        wei = POS_W[ipos]
    else:
        wei = random.choice(list(POS_W.values()))
    return(wei)            
#if a vector exists in the model, its POS weight is selected, else one random POS weight is selected

#===============================================================================
# COMMON
#===============================================================================
def cosine_similarity(v1, v2):
    if not numpy.any(v1) or not numpy.any(v2): return(0.0) #in case there is an empty vector we return 0.0
    cos_sim = 1.0 - round(spatial.distance.cosine(v1, v2), PRECISION_COS)
    #if math.isnan(cos_dist): cos_dist = 0.0  #just to avoid NaN on the code-output for the cosine-dist value -  some iword vectors might be 0.0 for all dimensions
    return (cos_sim)
#distance.cosine for v1 and v2 with precision of PRECISION_COS
#spatial.distance.cosine(v1, v2) gives cosine between them; we want their similarity - so 1 - cos(theta)
#it will return 0.0 for any empty vector received

def validateDocumentToken(doc_chain):
    if(doc_chain):
        return (True)
    else:
        return (False)  
#return false in case document has no word-token-synset

#===========================================================================
# NOT USED
#===========================================================================
def initialize_weights(dimensions):
    weight_constants = []
    for key in POS_W:
        tmp_dim = np.full(dimensions,POS_W[key])
        weight_constants.append(tmp_dim)
    return(weight_constants)
#initialize the weights based on the number of dimensions our model has and the constants in POS_W    
#POS_TAG: 0 ->  noun (n); 1-> verbs (v); 2->adverbs(r); 3->adjectives (a or s)

def select_weight(pos_tag):
    tag_index = 0
    if pos_tag == 'n':
        tag_index = 0
    elif tag_index == 'v':
        tag_index = 1
    elif tag_index == 'r':
        tag_index = 2
    else:#basically 'a' or 's'
        tag_index = 3
    return(tag_index)    
#defines a index number based on POS_TAG: 0 ->  noun (n); 1-> verbs (v); 2->adverbs(r); 3->adjectives (a or s)




#================================
def hypernyms_path(synset):
    hyps = lambda s:s.hypernyms()
    return(list(synset.closure(hyps))) 
#Return the transitive closure of source under the rel-  relationship, breadth-first
#returns a list of hypernyms to the root - better used in NOUNS

def matching_hypernyms(synset, other):
    matches = list()
    setas = hypernyms_path(synset)
    setbs = hypernyms_path(other)
    
    for a in setas:
        for b in setbs:
            if (a == b) and (a not in matches):
                matches.append(a)
            else:
                pass
    return(matches) #remember to check if list is empty
#returns a list of matches in two closures of synsets
 
#===============================================================================
# a = wn.synsets('fire')[0] # @UndefinedVariable
# b = wn.synsets('water')[0] # @UndefinedVariable
# print(a.lowest_common_hypernyms(b))
# x = matching_hypernyms(a, b)
# print(x)
#===============================================================================

#===============================================================================
# print(a.common_hypernyms(b))
# print(a.lowest_common_hypernyms(b))
# a = hypernyms_path(a)
# b = hypernyms_path(b)
# print()
# print(set(a).intersection(b))
# print(list(set(a).intersection(set(b))))
# print()
# print(a)
# print(b)
#===============================================================================

#build_synset_relations(c.ioffset(), c.ipos()) #'n',6901053