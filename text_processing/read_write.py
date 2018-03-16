'''
Created on Mar 16, 2018

@author: Terry Ruas
'''
#general imports
import os

#local imports
from lexicon import token_data

#input-folder:
doc_list_name = 'BSID_doclist.txt'
corpus_bsd = 'bsd_corpus'


def process_token(files):
    for file in files:
        tokens_list = []
        print('Processing %s' %file)
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                block = line.split('\t')
                #block[0] - word; block[1]-synset; block[2]-offset; block[3]-pos
                token = token_data(block[0],block[1],block[2],block[3].strip('\n'))
                tokens_list.append(token)
    return(tokens_list)
#creates a list of tokens for each document. A token is composed by: word, synset, offset and pos

def process_one_file(files, output_folder):
    big_document = open(output_folder+'/'+corpus_bsd, 'w+')    
    for file in files:
        print('Processing %s' %file)
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                block = line.split('\t')
                #block[0]:word; block[1]:synset; block[2]:offset; block[3]:pos - this has \n at the end
                big_document.write(block[0] +'-'+ block[2] +'-'+ block[3].strip('\n') + '\t')
        big_document.write('\n')
    big_document.close()   
#creates one file with each line being a document in the files list

def doclist_multifolder(folder_name):
    input_file_list = []
    for roots, dir, files in os.walk(folder_name):
        for file in files:
            file_uri = os.path.join(roots, file)
            #file_uri = file_uri.replace("\\","/") #if running on windows           
            if file_uri.endswith('txt'): input_file_list.append(file_uri)
    return input_file_list
#creates list of documents in many folders

def process_many_files(files, input_folder, output_folder):
    names = os.listdir(input_folder)   
    for index, file in enumerate(files):
        big_document = open(output_folder+'/'+names[index], 'w+')
        print('Processing %s' %file)
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                block = line.split('\t')
                #block[0]:word; block[1]:synset; block[2]:offset; block[3]:pos - this has \n at the end
                big_document.write(block[0] + '\t' + block[2] +'-'+ block[3].strip('\n') + '\t')
        big_document.write('\n')
        big_document.close()   
#creates one file per document parsed - clean features -> block[x]          





#===============================================================================
# ins = 'C:/tmp_project/LexicalChain_Builder'
# ons = 'C:/tmp_project/LexicalChain_Builder/input'
# x = os.listdir(ins)
# y = os.listdir(ons)
# 
# print(x)
# print(y)
#===============================================================================