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
- In *read_write.py* under *def fname_splitter(docslist)* If running in UNIX use *split('/')* if running in WINDOWS with 'hardcoded' path for input/output use *split('\\')*
- Takes directory with synsets in .txt files with the following format:

	word \t synset \t offset \t token4 \n

Example:
	gray	Synset('gray.n.09')	11012474	n


- Produces as many files as the input, also in the same format. However, each entry represents now a *LexicalChain*.
- POS_W = {'n':1.0, 'v':1.0, 'r':1.0, 'a':1.0, 's':1.0} in *lc_management.py* represent the weight for each POS when deciding which is closer to the average in that chain. This can be adjusted according to the distribution os POS in the trained corpus. Current is 1.0 for all.
 - 'a' and 's' should have the same value since both represent ADJECTIVES in WordNet

COMMAND LINE
=============
	python3 lc_builder.py  --input <input_folder> --chain <chain_type> [--size <size>] --output <output_folder> --model <model_file>
	
- <input_folder> : Input folder with .txt files or folders with .txt
- <chain_type> : 'flex' for Flexible Lexical Chains (FLLC); 'fixed' - for Fixed Lexical Chains (FXLC)
- <size> : [OPTIONAL] - size of the chunk for fixed chains (Default = CHUNK_SIZE in lc_management.py)
- <output_folder>: Ouput folder where LexicalChain representatives should be saved
- <model_file>: Synset-Embbedding model used. This should be in .vector format, but it can be changed to binary. The important is that its embeddings should be trained using synsets in the following canonical format: *word#offset#pos* . These are the keys to look up the embeddings.
- input/output/model folder must be in the same level as ../lc_builder.py (a level above the executed script)

Models and Corpora:
==============
All datasets, training corpora and generated models for the paper "_Enhanced word embeddings using multi-semantic representation through lexical chains_" 
can be found at [DeepBlue repository](https://deepblue.lib.umich.edu/data/concern/data_sets/w9505046h?locale=en)

UPDATES
=======
[2019-05-15]
1. Public domain for datasets/vectors/models generated.

[2019-03-07]
1. Moving project from personal repository

[2019-01-12]
1. Bug correction - reading non ASCII chars handled

[2018-11-29]:
1. General refactorin on printing status (reduce I/O)
2. Differences between python 3.4<= and 3.5>= with respect to merge dictionaries 
3. Discard documents that cannot be parsed into chains and/or are empty

[2018-11-15]:
1. Flex and Fixed LC implemented, IDE and command line - milestone
2. Small refactoring to validate input/parameteres
3. General refactoring in the code

[2018-11-14]
1. Flexible Lexical Chains (FLLC) - Prototype working
2. Fixed Lexical Chains (FXLC) - Prototype working - milestone

[2018-11-08]
1. Refactoring - work with document structure better
2. Refactoring - generating key and model handling
3. Refactoring - normal distribution between LOW-HIGH in case key does not exist in vector model
4. Initialize FixedLexical Chains
5. Making code for representing Fixed and Flex chains more common so they can share unit-simple functions. 

[2018-10-11]
1. If key-token does not exist on token-embeddings models, we generate a random uniform distribution [-5.0,5.0]. A random part-of-speech weight is also selected from the weight list
	1a. This shouldn't happen since the model used here is based on the synset-corpus we use to build the chains
2. General refactor for optimization
3. on doc_multifolder : file_uri = file_uri.replace("\\","/") #if running on windows
4. included new related synset methods ('topic_domains', 'region_domains', 'usage_domains')

[2018-06-12]
1. Deleted package for read-write. Everything will be under lexicon package

[2018-03-28]
1. Flexible Lexical Chain Algorithm  (FLC) implemented and validated.