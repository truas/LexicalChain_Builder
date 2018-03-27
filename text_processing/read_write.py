'''
Created on Mar 16, 2018

@author: Terry Ruas
'''
#general imports
import os


#local imports
from lexicon import token_data

#input-folder:



def process_token(files):
    for file in files:
        tokens_list = []
        print('Processing %s' %file)
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                block = line.split('\t')
                #block[0] - word; block[1]-synset; block[2]-offset; block[3]-pos
                token = token_data(block[0], block[1], block[2], block[3].strip('\n'))
                tokens_list.append(token)
    return(tokens_list)
#creates a list of tokens for each document. A token is composed by: word, synset, offset and pos


def doclist_multifolder(folder_name):
    input_file_list = []
    for roots, dir, files in os.walk(folder_name):
        for file in files:
            file_uri = os.path.join(roots, file)
            #file_uri = file_uri.replace("\\","/") #if running on windows           
            if file_uri.endswith('txt'): input_file_list.append(file_uri)
    return input_file_list
#creates list of documents in many folders

def chain_ouput_file(chains, bsd_fname, bsd_folder):  
    #print('Saving %s Document' %bsd_fname)
    doc_chain = open(bsd_folder +'/'+ bsd_fname, 'w+')  
    #currently using just Word \t SynsetID \t offset  \t pos
    for chain in chains:
        doc_chain.write(chain.iword + '\t' + chain.isyn + '\t' + chain.ioffset + '\t' + chain.ipos + '\n')
    doc_chain.close()
    #print('%s Document saved' %bsd_fname)  
#save each document(word, synset, offset, pos)         


#===============================================================================
# ins = 'C:/tmp_project/LexicalChain_Builder'
# ons = 'C:/tmp_project/LexicalChain_Builder/input'
# x = os.listdir(ins)
# y = os.listdir(ons)
# 
# print(x)
# print(y)
#==============================================================================