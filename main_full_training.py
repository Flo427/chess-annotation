import nltk
import re
import time
import random

MIN_COMMENT_LENGTH = 3
MAX_COMMENT_LENGTH = 49
MIN_ENGLISH_WORDS = 3
LOWERCASE = True
LEMMATIZING = True
MIN_FREQ = 5
MAX_FEATURES = 2000
SET_SIZE = 2000

files = [
	'bali02.pgn',
	'chessdoctor.pgn',
	'd00_chess_informant.pgn',
	'electronic_campfire.pgn',
	'europe_echecs.pgn',
	'exeter_lessons_from_tal.pgn',
	'famous_games.pgn',
	'GM_games.pgn',
	'great_masters.pgn',	
	'hartwig.pgn',
	'hayes.pgn',
	'human_computer.pgn',
	'immortal_games.pgn',
	'kasp_top.pgn',
	'kk.pgn',
	'koltanowski.pgn',
	'kramnik.pgn',
	'linares_2001.pgn',
	'linares_2002.pgn',
	'middleg.pgn',
	'moscow64.pgn',
	'newyork1924.pgn',
	'perle.pgn',
	'polgar.pgn',
	'pon_korch.pgn',
	'romero.pgn',
	'russian_chess.pgn',
	'scarborough_2001.pgn',
	'scca.pgn',
	'schiller.pgn',
	'semicomm.pgn',
	'top_games.pgn',
	'vc_89_99.pgn',
	'vc_2000.pgn',
	'vc_2001.pgn',
	'wijk_2003_annotated.pgn',
	'wijk_2004_annotated.pgn',
	'world_matches.pgn',
	'chessbasedb.pgn',
]

output_class_total = {
	"1": 1,
	"2": 1,
	"3": 1,
	"4": 1,
	"5": 1,
	"6": 1,
	"!": 1,
	"?": 1,
	"!!": 1,
	"??": 1,
	"!?": 1,
	"?!": 1,
	"7": 1,
	"8": 1,
	"10": 2,
	"11": 2,
	"12": 2,
	"13": 2,
	"14": 2,
	"15": 2,
	"16": 2,
	"17": 2,
	"18": 2,
	"19": 2,
	"22": 2,
	"32": 2,
	"36": 2,
	"40": 2,
	"44": 2,
	"132": 2,
	"133": 2,
	"136": 0,
	"138": 0,
	"140": 0,
	"141": 0,
	"142": 0,
	"143": 0,
	"144": 0,
	"145": 0,
	"146": 0,
}

output_class_move_1 = {
	"1": 1,
	"2": 2,
	"3": 1,
	"4": 2,
	"5": 1,
	"6": 2,
	"!": 1,
	"?": 2,
	"!!": 1,
	"??": 2,
	"!?": 1,
	"?!": 2,
	"7": 0,
	"8": 0,
	"10": 0,
	"11": 0,
	"12": 0,
	"13": 0,
	"14": 0,
	"15": 0,
	"16": 0,
	"17": 0,
	"18": 0,
	"19": 0,
	"22": 0,
	"32": 0,
	"36": 0,
	"40": 0,
	"44": 0,
	"132": 0,
	"133": 0,
	"136": 0,
	"138": 0,
	"140": 0,
	"141": 0,
	"142": 0,
	"143": 0,
	"144": 0,
	"145": 0,
	"146": 0,
}

output_class_move_2 = {
	"1": 2,
	"2": 5,
	"3": 1,
	"4": 6,
	"5": 3,
	"6": 4,
	"!": 2,
	"?": 5,
	"!!": 1,
	"??": 6,
	"!?": 3,
	"?!": 4,
	"7": 0,
	"8": 0,
	"10": 0,
	"11": 0,
	"12": 0,
	"13": 0,
	"14": 0,
	"15": 0,
	"16": 0,
	"17": 0,
	"18": 0,
	"19": 0,
	"22": 0,
	"32": 0,
	"36": 0,
	"40": 0,
	"44": 0,
	"132": 0,
	"133": 0,
	"136": 0,
	"138": 0,
	"140": 0,
	"141": 0,
	"142": 0,
	"143": 0,
	"144": 0,
	"145": 0,
	"146": 0,
}

output_class_position_1 = {
	"1": 0,
	"2": 0,
	"3": 0,
	"4": 0,
	"5": 0,
	"6": 0,
	"!": 0,
	"?": 0,
	"!!": 0,
	"??": 0,
	"!?": 0,
	"?!": 0,
	"7": 0,
	"8": 0,
	"10": 2,
	"11": 2,
	"12": 2,
	"13": 2,
	"14": 1,
	"15": 3,
	"16": 1,
	"17": 3,
	"18": 1,
	"19": 3,
	"22": 0,
	"32": 0,
	"36": 0,
	"40": 0,
	"44": 0,
	"132": 0,
	"133": 0,
	"136": 0,
	"138": 0,
	"140": 0,
	"141": 0,
	"142": 0,
	"143": 0,
	"144": 0,
	"145": 0,
	"146": 0,
}

output_class_position_2 = {
	"1": 0,
	"2": 0,
	"3": 0,
	"4": 0,
	"5": 0,
	"6": 0,
	"!": 0,
	"?": 0,
	"!!": 0,
	"??": 0,
	"!?": 0,
	"?!": 0,
	"7": 0,
	"8": 0,
	"10": 4,
	"11": 4,
	"12": 4,
	"13": 4,
	"14": 3,
	"15": 5,
	"16": 2,
	"17": 6,
	"18": 1,
	"19": 7,
	"22": 0,
	"32": 0,
	"36": 0,
	"40": 0,
	"44": 0,
	"132": 0,
	"133": 0,
	"136": 0,
	"138": 0,
	"140": 0,
	"141": 0,
	"142": 0,
	"143": 0,
	"144": 0,
	"145": 0,
	"146": 0,
}

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from langid.langid import LanguageIdentifier, model

start = time.time()
random.seed(0)

tokenizer = RegexpTokenizer('#[\w\d]{2}|\$\d+|[!\?]+|[\-\+/=]+|1/2|(?:\w\.)+|\.+|[\w\d\-\']+|\S')
lemmatizer = WordNetLemmatizer()
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

"""
Extract comments of .pgn-file
"""

print("\nEXTRACT COMMENTS OF PGN-FILE\n")

def read_comments_of_file(file, comment_data):
	raw = open(file, 'rb').read()
	string = raw.decode('iso-8859-1')
	
	pairs = re.findall(r'\$(?P<class>[0-9]+)\s*\{(?P<comment>[^{}]*)\}', string)		#NAG
	pairs += re.findall(r'\$(?P<class>[0-9]+)\s*\$[0-9]+\s*\{(?P<comment>[^{}]*)\}', string)		#NAG
	pairs += re.findall(r'(?P<class>[!\?]{1,2})\s*\{(?P<comment>[^{}]*)\}', string)	#symbol
	
	global instances
	global instances_total
	for pair in pairs:
		instances_total += 1
		if not (instances_total % 1000):
			end = time.time()
			print(instances, instances_total, end - start)
		if LOWERCASE:
			comment = tokenizer.tokenize(pair[1].lower())
			if LEMMATIZING:
				comment = [lemmatizer.lemmatize(lemmatizer.lemmatize(token, 'n'), 'v') for token in comment]
		else:
			comment = tokenizer.tokenize(pair[1])
		if len(comment) < MIN_COMMENT_LENGTH or len(comment) > MAX_COMMENT_LENGTH:
			continue
		test = identifier.classify(pair[1])
		if test[0] != "en" and test[1] > 0.99:
			continue
		if output_class_total[pair[0]]:
			count = 0
			for token in comment:
				if (lemmatizer.lemmatize(token, 'n') in english_vocab or lemmatizer.lemmatize(token, 'v') in english_vocab) and count == MIN_ENGLISH_WORDS:
					break
				if (lemmatizer.lemmatize(token, 'n') in english_vocab or lemmatizer.lemmatize(token, 'v') in english_vocab) and count < MIN_ENGLISH_WORDS:
					count += 1
			if count == MIN_ENGLISH_WORDS:
				instances += 1
				comment_data += [(comment, pair[0])]
					
comment_data = []
instances = 0
instances_total = 0
for file in files:
	read_comments_of_file('files/' + file, comment_data)

random.shuffle(comment_data)
comment_data_set_1 = comment_data[:]
comment_data.reverse()
comment_data_set_2 = [comment for comment in comment_data if len(comment[0]) < 10]
comment_data_set_3 = [comment for comment in comment_data if len(comment[0]) > 9 and len(comment[0]) < 50]

comment_data_set_1_move = [comment for comment in comment_data_set_1 if output_class_move_1[comment[1]]]
comment_data_set_2_move = [comment for comment in comment_data_set_2 if output_class_move_1[comment[1]]]
comment_data_set_3_move = [comment for comment in comment_data_set_3 if output_class_move_1[comment[1]]]

comment_data_set_1_position = [comment for comment in comment_data_set_1 if output_class_position_1[comment[1]]]
comment_data_set_2_position = [comment for comment in comment_data_set_2 if output_class_position_1[comment[1]]]
comment_data_set_3_position = [comment for comment in comment_data_set_3 if output_class_position_1[comment[1]]]

comment_data_set_1 = comment_data_set_1_move[:SET_SIZE] + comment_data_set_1_position[:SET_SIZE]
comment_data_set_2 = comment_data_set_2_move[:SET_SIZE] + comment_data_set_2_position[:SET_SIZE]
comment_data_set_3 = comment_data_set_3_move[:SET_SIZE] + comment_data_set_3_position[:SET_SIZE]

end = time.time()
print(instances, instances_total, end - start)

"""
Ectract classification features
"""

print("\nEXTRACT CLASSIFICATION FEATURES\n")

import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import coo_matrix

def identity_tokenizer(text):
    return text

comments_set = [comment for (comment,_) in comment_data]
comments_set_1 = [comment for (comment,_) in comment_data_set_1]
comments_set_2 = [comment for (comment,_) in comment_data_set_2]
comments_set_3 = [comment for (comment,_) in comment_data_set_3]

"""
count
"""

count_matrix = {}
count_feature_names = {}
for (data_set, comments) in [("set-1", comments_set_1), ("set-2", comments_set_2), ("set-3", comments_set_3)]:
	count_vectorizer = CountVectorizer(min_df=MIN_FREQ, max_features=MAX_FEATURES, ngram_range=(1,3), tokenizer=identity_tokenizer, lowercase=False)
	count_vectorizer.fit(comments_set)
	count_matrix[data_set] = count_vectorizer.transform(comments)
	count_feature_names[data_set] = count_vectorizer.get_feature_names()

end = time.time()
print("count finished " + str(end - start))

"""
tf-idf
"""

tfidf_matrix = {}
tfidf_feature_names = {}
for (data_set, comments) in [("set-1", comments_set_1), ("set-2", comments_set_2), ("set-3", comments_set_3)]:
	tfidf_vectorizer = TfidfVectorizer(min_df=MIN_FREQ, max_features=MAX_FEATURES, ngram_range=(1,3), tokenizer=identity_tokenizer, lowercase=False)
	tfidf_vectorizer.fit(comments_set)
	tfidf_matrix[data_set] = tfidf_vectorizer.transform(comments)
	tfidf_feature_names[data_set] = tfidf_vectorizer.get_feature_names()

end = time.time()
print("tf-idf finished " + str(end - start))

"""
word embeddings
"""

def build_matrix(model, comments):
	matrix = np.zeros(shape=(len(comments), model.wv.vector_size))
	for idx, comment in enumerate(comments):
		vectors = [model.wv[token] * tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[token]] for token in comment if token in model.wv.vocab.keys()]
		matrix[idx] = np.mean(vectors, axis=0)
	return matrix
	
w2v_own_matrix = {}
w2v_own_feature_names = {}
w2v_pretrained_matrix = {}
w2v_pretrained_feature_names = {}
model_own = Word2Vec(comments_set, min_count=MIN_FREQ)
model_own.train(comments_set, total_examples=len(comments_set), epochs=10)
model_own.save("chess-annotations.model")
model_pretrained = model_own#KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

for (data_set, comments) in [("set-1", comments_set_1), ("set-2", comments_set_2), ("set-3", comments_set_3)]:
	tfidf_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
	tfidf_vectorizer.fit(comments_set)
	w2v_own_matrix[data_set] = build_matrix(model_own, comments)
	w2v_own_feature_names[data_set] = range(1, w2v_own_matrix[data_set].shape[1] + 1)
	w2v_pretrained_matrix[data_set] = build_matrix(model_pretrained, comments)
	w2v_pretrained_feature_names[data_set] = range(1, w2v_pretrained_matrix[data_set].shape[1] + 1)

end = time.time()
print("word embeddings finished " + str(end - start))

"""
Write arff files
"""

print("\nWRITE ARFF FILES\n")

def write_arff(comment_data, data_set, feature_type, problem_name, classes, output_class, feature_names, matrix):
    arff = open("chess-annotations-" + data_set + "-" + feature_type + "-" + problem_name + ".arff", "w")
    RELATION_NAME = "comment-" + problem_name									 
    buff = ("@RELATION " + RELATION_NAME + "\n")
    buff += ("@ATTRIBUTE number_of_tokens INTEGER\n") 
    for feature in feature_names:
        buff += ("@ATTRIBUTE " + feature_type + "(" + str(feature).replace(" ", "_").replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_") + ") REAL\n")
    buff += ("@ATTRIBUTE class {" + classes + "}\n") 
    buff += ("@DATA\n")    
    for comment in range(len(comment_data)):
        if not output_class[comment_data[comment][1]]:
            continue
        row = coo_matrix(matrix[comment,:])
        row = sorted(zip(row.col, row.data), key=lambda x: (x[0]))
        buff += ("{0 " + str(len(comment_data[comment][0])) + ", ")
        for feature, value in row:            
            buff += (str(feature + 1) + " " + str(value) + ", ")
        buff += (str(len(feature_names) + 1) + " " + str(output_class[comment_data[comment][1]]) + "}\n")        
    arff.write(buff)

for (data_set, comment_data) in [("set-1", comment_data_set_1), ("set-2", comment_data_set_2), ("set-3", comment_data_set_3)]:
	for (feature_type, feature_names, matrix) in [("count", count_feature_names, count_matrix), ("tfidf", tfidf_feature_names, tfidf_matrix), ("w2v-own", w2v_own_feature_names, w2v_own_matrix), ("w2v-pretrained", w2v_pretrained_feature_names, w2v_pretrained_matrix)]:
		write_arff(comment_data, data_set, feature_type, "total", "1, 2", output_class_total, feature_names[data_set], matrix[data_set]) 
		write_arff(comment_data, data_set, feature_type, "move-1", "1, 2", output_class_move_1, feature_names[data_set], matrix[data_set])  
		write_arff(comment_data, data_set, feature_type, "move-2", "1, 2, 3, 4, 5, 6", output_class_move_2, feature_names[data_set], matrix[data_set])  
		write_arff(comment_data, data_set, feature_type, "position-1", "1, 2, 3", output_class_position_1, feature_names[data_set], matrix[data_set])  
		write_arff(comment_data, data_set, feature_type, "position-2", "1, 2, 3, 4, 5, 6, 7", output_class_position_2, feature_names[data_set], matrix[data_set])  

end = time.time()
print(end - start)
