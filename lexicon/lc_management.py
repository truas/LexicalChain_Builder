'''
Created on Mar 19, 2018

@author: Terry Ruas

'''
#imports
import numpy
import random
import numpy as np
from scipy import spatial
from nltk.corpus import wordnet as wn

#local-import
from lexicon import token_data as td



#Global - Definitions
PRECISION_COS = 7


#definitions

#synset node relationships in WordNet
NYMS = ['hypernyms','instance_hypernyms' , 'hyponyms', 'instance_hyponyms', 
        'member_holonyms', 'substance_holonyms', 'part_holonyms', 
        'member_meronyms', 'substance_meronyms', 'part_meronyms',
        'attributes','entailments','causes', 'also_sees', 'verb_groups', 'similar_tos',
        'topic_domains', 'region_domains', 'usage_domains']

#Part od speech weights
POS_W = {'n':1.0, 'v':1.0, 'r':1.0, 'a':1.0, 's':1.0}

#===============================================================================
# FlexChain Block - START
#===============================================================================
def build_FlexChain(data_tokens, vec_model):
    #initialize chain
    flex_chains = []
    flex_chains.append(start_FlexChain(data_tokens[0])) #the first element is already in the chain    
    
    for counter, token in enumerate(data_tokens,1):#visit every token except the first   
        adopt = False
        #last element of the chain
        last = len(flex_chains)-1
        tmp_iddata = td.idData()
        #tmp chains building block
        tmp_iddata.iword = token.word 
        tmp_iddata.isyn = wn.synset_from_pos_and_offset(token.pos, token.offset)  # @UndefinedVariable
        tmp_iddata.ioffset = token.offset        
        tmp_iddata.ipos = token.pos
        tmp_ssr = build_synset_relations(tmp_iddata.ioffset, tmp_iddata.ipos)
        
        #if there was a match between both SSR we will add this synset to the chain 
        for key in set(tmp_ssr) & set(flex_chains[last].chain_relation_tokens):
            if tmp_ssr[key] == flex_chains[last].chain_relation_tokens[key]:
                adopt = True
                break 
        
        if adopt:#incorporate synset into current chain
            flex_chains[last].chain_relation_tokens = {**tmp_ssr, **flex_chains[last].chain_relation_tokens} #merge SSR between chain and synset
            flex_chains[last].prospective_tokens.append(tmp_iddata)
        else:#calculate current chain representative and create a new one to start a new chain block
            flex_chains[last].chain_id = represent_FlexChain(flex_chains[last], vec_model)
            new_chain =  start_FlexChain(token)
            flex_chains.append(new_chain) 

    return(flex_chains)
#build flexible lexical chain

def represent_FlexChain(current_chain, vec_model):
    vecs = []
    weight_pos =[]
    
    if len(current_chain.prospective_tokens) is not 1:        
        for item in current_chain.prospective_tokens:
            try:
                key = item.iword+'#'+str(item.ioffset)+'#'+item.ipos
                vec = vec_model.word_vec(key) #return the vector for the token in the model
                vecs.append(vec) #make a list of all token-vector from the synset embedding
                wei = POS_W[item.ipos]
                weight_pos.append(wei)#create a weigth vector based on the pos_tag
            except KeyError:#in case key is not on the model, create array (model-size) with random uniform dist, from -1 to 1
                vecs.append(np.random.uniform(low=-1.0, high=1.0, size = vec_model.vector_size))
                weight_pos.append(random.choice(list(POS_W.values())))#get a random weight based on the POS weight values
        
        current_chain_avg = np.average(vecs, weights = weight_pos, axis=0) #weighted average of the current chain - weights are the values on POS_W
        chain_rep = closest_synset_rep(current_chain.prospective_tokens, current_chain_avg, vec_model) 
    else:
        chain_rep = current_chain.chain_id #if there is one element in the chain, that's its ID
        #this is already done when starting a new chain
            
    return(chain_rep)

def closest_synset_rep(prospective_ids, chain_avg, vec_model):
    highest_sofar = -float('inf')
    choice = 0
    
    for i, candidates in enumerate(prospective_ids):
        key = candidates.iword+'#'+str(candidates.ioffset)+'#'+candidates.ipos
        #in case key is not on the mode build an empty array
        try:
            v2 = vec_model.word_vec(key)
        except KeyError:#if one of the synsets are not in the model we set is as [0.0] so the cosine_sim returns 0.0 directly
            v2 = [0.0] #np.full(vec_model.syn0[0].size, 0.0)  #np.full(vec_model.syn0[0].size, 0.0)
        
        tmp = cosine_similarity(chain_avg, v2)
        #keep the index of the element with the highest cos-sim with the average in the chain
        if tmp >= highest_sofar:
            highest_sofar = tmp
            choice = i
            
    return(prospective_ids[choice]) #an idData will be returned to represent the current chain   
#elects the idData(synset-key) with the highest cosine against the average of current chain prospective synsets            

def start_FlexChain(data_token):
    chain_start = td.ChainData()
    chain_iddata = td.idData()
    #chain possible representative
    chain_iddata.iword = data_token.word
    chain_iddata.isyn = wn.synset_from_pos_and_offset(data_token.pos, data_token.offset)  # @UndefinedVariable
    chain_iddata.ioffset = data_token.offset
    chain_iddata.ipos = data_token.pos
    chain_start.prospective_tokens.append(chain_iddata)
    #chains related synsets
    chain_start.chain_relation_tokens = build_synset_relations(chain_iddata.ioffset, chain_iddata.ipos)
    #chain id  setup
    chain_start.chain_id = chain_iddata
    
    return(chain_start)
#initializes first element of a chain  and set this chains id with the first token (word#offset#pos)

def build_synset_relations(offset, pos):
    relation_synsets = dict()
    synset = wn.synset_from_pos_and_offset(pos, offset) # @UndefinedVariable
    relation_synsets[synset] = 1 #the synset itself is part of the related ones
        
    for nym in NYMS:
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
#produces a list of synsets from all the *_NYMS from that synset - produces SSR from a synset(offset,pos)


#===============================================================================
# FlexChain Block - END
#===============================================================================

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


def cosine_similarity(v1, v2):
    if not numpy.any(v1) or not numpy.any(v2): return(0.0) #in case there is an empty vector we return 0.0
    cos_sim = 1.0 - round(spatial.distance.cosine(v1, v2), PRECISION_COS)
    #if math.isnan(cos_dist): cos_dist = 0.0  #just to avoid NaN on the code-output for the cosine-dist value -  some word vectors might be 0.0 for all dimensions
    return (cos_sim)
#distance.cosine for v1 and v2 with precision of PRECISION_COS
#spatial.distance.cosine(v1, v2) gives cosine between them; we want their similarity - so 1 - cos(theta)
#it will return 0.0 for any empty vector received

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

#build_synset_relations(c.offset(), c.pos()) #'n',6901053