import nltk
import re
import time
import random

MIN_COMMENT_LENGTH = 3
MAX_COMMENT_LENGTH = 49
MIN_ENGLISH_WORDS = 3
LOWERCASE = True
LEMMATIZING = False
MIN_FREQ = 1
MAX_FEATURES = 100000

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
#	'chessbasedb.pgn',
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
	str = raw.decode('iso-8859-1')
	
	pairs = re.findall(r'\$(?P<class>[0-9]+)\s*\{(?P<comment>[^{}]*)\}', str)		#NAG
	pairs += re.findall(r'\$(?P<class>[0-9]+)\s*\$[0-9]+\s*\{(?P<comment>[^{}]*)\}', str)		#NAG
	pairs += re.findall(r'(?P<class>[!\?]{1,2})\s*\{(?P<comment>[^{}]*)\}', str)	#symbol
	
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
end = time.time()
print(instances, instances_total, end - start)

"""
Ectract classification features
"""

print("\nEXTRACT CLASSIFICATION FEATURES\n")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import coo_matrix

comments = [" ".join(token) for (token,_) in comment_data]
count_vectorizer = CountVectorizer(min_df=MIN_FREQ, max_df=1.0, max_features=MAX_FEATURES, ngram_range=(1,3))
count_matrix = count_vectorizer.fit_transform(comments)
count_feature_names = count_vectorizer.get_feature_names()
tfidf_vectorizer = TfidfVectorizer(min_df=MIN_FREQ, max_df=1.0, max_features=MAX_FEATURES, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(comments)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

end = time.time()
print(end - start)

"""
Write arff files
"""

print("\nWRITE ARFF FILES\n")

def write_arff(feature_type, problem_name, classes, output_class, feature_names, matrix):
    arff = open("chess-annotations-" + feature_type + "-" + problem_name + ".arff", "w")
    RELATION_NAME = "comment-" + problem_name									 
    buff = ("@RELATION " + RELATION_NAME + "\n")
    buff += ("@ATTRIBUTE number_of_tokens INTEGER\n") 
    for feature in feature_names:
        buff += ("@ATTRIBUTE " + feature_type + "(" + str(feature).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_") + ") REAL\n")
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

for (feature_type, feature_names, matrix) in [("count", count_feature_names, count_matrix), ("tfidf", tfidf_feature_names, tfidf_matrix)]:
	write_arff(feature_type, "total", "1, 2", output_class_total, feature_names, matrix) 
	write_arff(feature_type, "move-1", "1, 2", output_class_move_1, feature_names, matrix)  
	write_arff(feature_type, "move-2", "1, 2, 3, 4, 5, 6", output_class_move_2, feature_names, matrix)  
	write_arff(feature_type, "position-1", "1, 2, 3", output_class_position_1, feature_names, matrix)  
	write_arff(feature_type, "position-2", "1, 2, 3, 4, 5, 6, 7", output_class_position_2, feature_names, matrix)  

end = time.time()
print(end - start)
