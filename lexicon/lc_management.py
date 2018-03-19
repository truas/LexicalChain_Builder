'''
Created on Mar 19, 2018

@author: Terry Ruas
'''
#imports
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import matching

#definitions
NYMS = ['hypernyms','instance_hypernyms' , 'hyponyms', 'instance_hyponyms', 
        'member_holonyms', 'substance_holonyms', 'part_holonyms', 
        'member_meronyms', 'substance_meronyms', 'part_meronyms',
        'attributes','entailments','causes', 'also_sees', 'verb_groups', 'similar_tos']#synset node relationships

def build_synset_relations(offset, pos):
    relation_synsets = dict()
    synset = wn.synset_from_pos_and_offset(pos, offset) # @UndefinedVariable
        
    for nym in NYMS:
        try:
            tmp_items = getattr(synset, nym)()
            for item in tmp_items:
                if item not in relation_synsets:
                    relation_synsets[item] = 0
                else:
                    continue
        except:
            continue        
                
    return (relation_synsets)
#produces a list of synsets from all the *_NYMS from that synset

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
            if a == b and a not in matches:
                matches.append(a)
            else:
                pass
    return(matches) #remember to check if list is empty
#returns a list of matches in two closures of synsets


#===============================================================================
# a = wn.synsets('dog')[0] # @UndefinedVariable
# b = wn.synsets('cat')[0] # @UndefinedVariable
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