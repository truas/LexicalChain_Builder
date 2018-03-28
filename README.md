# LexicalChain_Builder
- Works with Python 3.X, WordNet (native in nltk) and Synset-Embeddings (word embeddings model trained using synsets instead of words - synset2vec.vector)
- Another type of synset2vec can be used, but changes have to be made to retrieve key-vectors in the model
- Transforms consecutive related synsets (semantic related synsets) into a LexicalChains by incorporating in the same chain synsets that share one or more of the following attributes/relationships defined in WordNet as:

	NYMS = ['hypernyms','instance_hypernyms' , 'hyponyms', 'instance_hyponyms', 
        'member_holonyms', 'substance_holonyms', 'part_holonyms', 
        'member_meronyms', 'substance_meronyms', 'part_meronyms',
        'attributes','entailments','causes', 'also_sees', 'verb_groups', 'similar_tos'] + [evaluated synset]

- [evaluated synset] is the synset-token being evaluated itself     
- These chains grow as long there are semantic related synsets in common
- In *text_processing.read_write.py* under *def fname_splitter(docslist)* If running in UNIX use *split('/')* if running in WINDOWS with 'hardcoded' path for input/output use *split('\\')*
- Takes directory with synsets in .txt files with the following format:

	word \t synset \t offset \t token4 \n

Example:
	gray	Synset('gray.n.09')	11012474	n


- Produces as many files as the input, also in the same format. However, each entry represents now a *LexicalChain*.

COMMAND LINE
=============
	python3 lc_builder.py  --input <input_folder> --ouput <output_folder> --model <model_file>
	
- <input_folder> : Input folder with .txt files or folders with .txt
- <output_folder>: Ouput folder where LexicalChain representatives should be saved
- <model_file>: Synset-Embbedding model used. This should be in .vector format, but it can be changed to binary. The important is that its embeddings should be trained using synsets in the following canonical format: *word#offset#pos* . These are the keys to look up the embeddings.
- input/output/model folder must be in the same level as ../lc_builder.py (a level above the executed script)


UPDATES
=======
[2018-03-28]
1. Flexible Lexical Chain Algorithm  (FLC) implemented and validated.