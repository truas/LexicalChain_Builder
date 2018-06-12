'''
Created on Mar 16, 2018

@author: Terry Ruas
IMPORTAT:
1. The whole manipulation of offsets is done with an INT, during read/write we should parse it to INT/STRING accordingly
2. The offset needs to be an INT to use 'wn.synset_from_pos_and_offset(pos,offset)'

'''
#import
import logging
import gensim
import time
import nltk
import sys
import argparse #for command line arguments
import os

#from-imports
from datetime import timedelta
from stop_words import get_stop_words


#python module absolute path
pydir_name = os.path.dirname(os.path.abspath(__file__))

#python path definition
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

#local imports
from lexicon import read_write as rw
from lexicon import token_data as td
from lexicon import lc_management as lm

#input/output files/folder - If you need to set input, output and model folders
#in_foname = 'C:/tmp_project/LexicalChain_Builder/input'
#ou_foname = 'C:/tmp_project/LexicalChain_Builder/output'
#mo_foname = 'C:/tmp_datasets/Wikipedia_Dump/2018_01_20/models/300d-neg-5w-5mc-cbow/wikidump2080120-nbsd-300d-neg-5w-5mc-cbow.vector'
#mo_foname = "C:/tmp_datasets/Wikipedia_Dump/word2vec_gensim_wiki/wiki.en.text.vector" #binary-false


tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
nltk.download('wordnet') #just to guarantee wordnet from nltk is installed

#overall runtime start
start_time = time.monotonic()

#Main core
if __name__ == '__main__':  
    #show logs
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
     
    #IF you want to use COMMAND LINE for folder path
    parser = argparse.ArgumentParser(description="BSD_Extractor - Transforms text into synsets")
    parser.add_argument('--input', type=str, action='store', dest='inf', metavar='<folder>', required=True, help='input folder to read document(s)')
    parser.add_argument('--output', type=str, action='store', dest='ouf', metavar='<folder>', required=True, help='output folder to write document(s)')
    parser.add_argument('--model', type=str, action='store', dest='mod', metavar='<folder>', required=True, help='trained word embeddings model')
    
    args = parser.parse_args()
     
    #COMMAND LINE  folder paths
    input_folder = args.inf
    output_folder = args.ouf
    model_folder = args.mod

    #in/ou relative location - #input/output/model folders are under synset/module/
    in_foname = os.path.join(pydir_name, '../'+input_folder) 
    ou_foname = os.path.join(pydir_name, '../'+output_folder)
    mo_foname = os.path.join(pydir_name, '../'+model_folder)
    
    #Loads
    trained_w2v_model = gensim.models.KeyedVectors.load_word2vec_format(mo_foname, binary=False) #If the model is not binary set binary=False
    #trained_w2v_model = gensim.models.KeyedVectors.load(mo_foname) #model.load used with .model extension - this files has to be in the same folder as its .npy
    
    synset_docslist = rw.doclist_multifolder(in_foname) #creates list of documents to parse
    synset_docsnames = rw.fname_splitter(synset_docslist) #name of the document
    counter = 0 #just to control the output file name
   
    for synset_docitem in synset_docslist:
        doc_data = td.DocumentData()
        
        doc_data.tokens = rw.process_token(synset_docitem)
        print('Document %s - Tokens Processed: %s'  %(synset_docsnames[counter],(timedelta(seconds= time.monotonic() - start_time))))
                
        doc_data.flex_chains = lm.build_FlexChain(doc_data.tokens, trained_w2v_model)
        print('Document %s - FlexChain Built: %s'  %(synset_docsnames[counter],(timedelta(seconds= time.monotonic() - start_time))))
        
        rw.chain_ouput_file(doc_data.flex_chains, synset_docsnames[counter], ou_foname)
        print('Document %s - Saved: %s'  %(synset_docsnames[counter],(timedelta(seconds= time.monotonic() - start_time))))
        
        counter+=1 
         
         
         
    print('finished...')
     



