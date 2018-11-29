#!/usr/bin/python3
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

this_dir = os.path.dirname(os.path.realpath(__file__))


#python module absolute path
pydir_name = os.path.dirname(os.path.abspath(__file__))
ppydir_name = os.path.dirname(pydir_name)

#python path definition
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


#local imports
from lexicon import lc_management as lm
from lexicon import token_data as td
from lexicon import read_write as rw 

#Extra models
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
    parser = argparse.ArgumentParser(description="Lexical Chain Builder - Transforms Word#synset#offset#POS into chains - in the same form")
    parser.add_argument('--input', type=str, action='store', dest='inf', metavar='<folder>', required=True, help='input folder to read document(s)')
    parser.add_argument('--chain', type=str, action='store', dest='chs', metavar='<parameter>', required=True, help='selects what kind of chain is used')
    parser.add_argument('--size', type=int, action='store', dest='csi', metavar='<parameter>', required=False, help='[optional] size of chain chunk for fixed chains')
    parser.add_argument('--output', type=str, action='store', dest='ouf', metavar='<folder>', required=True, help='output folder to write document(s)')
    parser.add_argument('--model', type=str, action='store', dest='mod', metavar='<folder>', required=True, help='trained word embeddings model')
       
    args = parser.parse_args()
        
    #COMMAND LINE  folder paths
    input_folder = args.inf
    output_folder = args.ouf
    model_folder = args.mod
    #print(input_folder, output_folder, model_folder)
   
     #in/ou relative location - #input/output/model folders are under synset/module/
    in_foname = os.path.join(ppydir_name, input_folder) 
    ou_foname = os.path.join(ppydir_name, output_folder)
    mo_foname = os.path.join(ppydir_name, model_folder)
    ch_select = rw.checkChainType(args.chs)#selects chain-type and size
    ch_size = args.csi#size of chunk - optional - (CHUNK_SIZE in lc_management)
    #print(in_foname,ou_foname,mo_foname,ch_select,ch_size)
 
    
    
    #===========================================================================
    # #IDE - Path Definitions
    # #input/output files/folder - If you need to set input, output and model folders
    # in_foname = 'C:/tmp_project/ChainBuilder/input_wd18/w001'
    # ch_select = False# = 'flex'# FLLC - flex (True); FXLC - fixed (False);
    # ch_size = '' #size of chain for fixed
    # ou_foname = 'C:/tmp_project/ChainBuilder/output'
    # mo_foname = 'C:/Users/terry/Documents/Datasets/Wikipedia_Dump/2018_01_20/models/300d-hs-15w-10mc-cbow.model'
    # #mo_foname = "C:/tmp_datasets/Wikipedia_Dump/word2vec_gensim_wiki/wiki.en.text.vector" #binary-false
    #===========================================================================

    
    #Loads
    #trained_w2v_model = gensim.models.KeyedVectors.load_word2vec_format(mo_foname, binary=False) #If the model is not binary set binary=False
    trained_w2v_model = gensim.models.KeyedVectors.load(mo_foname) #model.load used with .model extension - this files has to be in the same folder as its .npy
    
    synset_docslist = rw.doclist_multifolder(in_foname) #creates list of documents to parse
    synset_docsnames = rw.fname_splitter(synset_docslist) #name of the document
    
    status_check = 0 #only print every 5000 documents saved
    for counter,synset_docitem in enumerate(synset_docslist):
        doc_data = td.DocumentData()
         
        doc_data.tokens = rw.process_token(synset_docitem)
        #print('Document %s - Tokens Processed: %s'  %(synset_docsnames[counter],(timedelta(seconds= time.monotonic() - start_time))))
         
        if(ch_select):#FLLC
            doc_data.chains = lm.build_FlexChain(doc_data.tokens, trained_w2v_model)
            #print('Document %s - FlexChain Built: %s'  %(synset_docsnames[counter],(timedelta(seconds= time.monotonic() - start_time))))
        else:#FXLC
            doc_data.chains = lm.build_FixedChain(doc_data.tokens, trained_w2v_model, ch_size)
            #print('Document %s - FixedChain Built: %s'  %(synset_docsnames[counter],(timedelta(seconds= time.monotonic() - start_time))))    
         
        rw.chain_ouput_file(doc_data.chains, synset_docsnames[counter], ou_foname)
        if (status_check%5000==0): print('Document %s - Saved: %s'  %(synset_docsnames[counter],(timedelta(seconds= time.monotonic() - start_time)))) 
        status_check+=1  
    print('finished...')
     



