Author:		Florian Beck
Date:		2019/05/14

This repository contains all code, results and elaboration files 
in the context of the master thesis "Sentiment Classification of 
Chess Annotations" at TU Darmstadt. It is structured as follows:

-----------------------------------------------------------------
1 /code
-----------------------------------------------------------------

This folder contains the main.py, which is the only file to be 
executed to generate the data extraction of the pgn files and the 
writing of the arff files. It also generates the word embedding 
saved in the file chess-annotations.model and uses another word 
embedding based on Google News which is not included in the 
repository due to its size. However, it can be downloaded via:
https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM

The output will be 60 arff files named according to convention:
chess-annotations-set-X-MODEL-PROBLEM

X
1 := mixed-length comments
2 := short comments
3 := long comments

MODEL
count			:= count-based model
tfidf			:= TF-IDF-based model
w2v-own			:= chess annotations word embedding
w2v-pretrained	:= Google News word embedding

PROBLEM
move-1		:= move annotations (2 classes)
move-2		:= move annotations (6 classes)
position-1	:= position annotations (3 classes)
position-2	:= position annotations (7 classes)
total		:= move vs. position annotations (2 classes)

The configuration of the main.py can be customized by the 
following parameters:

Parameter				|Standard
------------------------+--------
MIN_COMMENT_LENGTH		|       3
MAX_COMMENT_LENGTH		|      49
SPLIT_COMMENT_LENGTH	|       9
MIN_ENGLISH_WORDS		|       3
LOWERCASE				|    True
LEMMATIZING				|    True
MIN_FREQ				|       5
MAX_FEATURES			|    2000
SET_SIZE				|    2000

MIN_COMMENT_LENGTH, MAX_COMMENT_LENGTH, SPLIT_COMMENT_LENGTH:
The first three parameters are used to filter the comments by 
their token length and implies MIN < SPLIT < MAX. The SPLIT 
parameter is used to separate short and long comments (set 2&3). 

MIN_ENGLISH_WORDS:
Filters the comments by the number of contained English words. 
Implies MIN_ENGLISH_WORDS <= MIN_COMMENT_LENGTH.

LOWERCASE & LEMMATIZING:
Boolean parameters to switch off preprocessing steps. 
LEMMATIZING will not have any effect if LOWERCASE = False.

MIN_FREQ:
Sets the minimal number of occurences of a term in all comments. 
Used in all models except of the pretrained word embedding.

MAX_FEATURES:
Limits the number of terms/features in the count- and TF-IDF-
based models.

SET_SIZE:
Limits the number of comments per set. The data set of the total 
problem contains all move and position comments and is twice as 
large.

The execution time in the standard configuration depends on the 
computing capacities; a rough benchmark is one hour.

-----------------------------------------------------------------
1.1 /code/archive
-----------------------------------------------------------------

This files have been used for testing and evaluations. They need 
to be put in the code folder to work properly.

The file create_w2v_model.py is a copy of main.py reduced to the 
creation of the chess-annotations.model.

The file model_tests.py is used to test the created word 
embedding with some basic queries.

The file token_count_statistics.py uses all comment data without 
any filters to calculate most frequent tokens and other 
statistics.

-----------------------------------------------------------------
1.2.1 /code/files/arff
-----------------------------------------------------------------

This folder contains all arff files used for the tests presented 
in the thesis (test_data_master_thesis) and the finally created 
arff files (final_data).

Note that new generated arff files will not overwrite them, but 
instead be stored in the code folder.

-----------------------------------------------------------------
1.2.2 /code/files/pgn
-----------------------------------------------------------------

This folder contains the used chess game files in PGN format. 
The chessbase.pgn is split into two parts due to its big size.

-----------------------------------------------------------------
1.2.3 /code/files/results
-----------------------------------------------------------------

This folder contains different results and statistics on which 
the tables and graphs in the thesis are based on. The most 
important results are stored in the set-1, set-2 and set-3 
folders; they store the evaluation of different classifiers. 
The accuracies are collected in the corresponding Excel file 
results-set-X. The best accuracies and the underlying 
configurations are stored in the file top-results.

The folder attributes contains evaluations of the attributes 
with the best information gain in different sets and problems 
for the count-based model.

The folder misclassification contains some evaluations of 
misclassified instances in the four models for the three problems 
move-1, position-1 and total. It also consideres the evaluation 
of the confusion matrices of the top results with different cost 
weights.

The folder tokenizer contains separate arff files and evaluations 
of different tokenizer configurations in the preprocessing.

The file file-statistics contains information about the NAG and 
symbol distribution in all pgn files.

The file token-statistics contains the information calculated by 
the file token_count_statistics.py.

The files w2v-own-results and w2v-pretrained-results store the 
results of the basic queries of model_tests.py.

-----------------------------------------------------------------
2 /texmf/doc/latex/tuddesign
-----------------------------------------------------------------

This folder contains the latex files and the pdf file of the 
thesis. All images are stored in the subfolder /images.

-----------------------------------------------------------------
3 /thesis
-----------------------------------------------------------------

This folder contains the contents of the thesis split into the 
chapters as well as the presentation of the thesis.