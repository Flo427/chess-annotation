import nltk
import re
import time
import random

MIN_COMMENT_LENGTH = 3
MAX_COMMENT_LENGTH = 49
MIN_ENGLISH_WORDS = 3
LOWERCASE = True
STEMMING = False
REMOVE_STOPWORDS = False
MIN_FREQ_WORD = 1
MIN_FREQ_BIGRAM = 1
MIN_FREQ_TRIGRAM = 1

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

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from langid.langid import LanguageIdentifier, model

start = time.time()
random.seed(0)

tokenizer = RegexpTokenizer('#[\w\d]{2}|\$\d+|[!\?]+|[\-\+/=]+|1/2|(?:\w\.)+|\.+|[\w\d\-\']+|\S')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

"""
Extract comments of .pgn-file
"""

print("\nEXTRACT COMMENTS OF PGN-FILE\n")

def read_comments_of_file(file, comment_data_total, comment_data_move, comment_data_position):
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
				comment_data_total += [(comment, pair[0])]
				if output_class_move_1[pair[0]]:
					comment_data_move += [(comment, pair[0])]
				if output_class_position_1[pair[0]]:
					comment_data_position += [(comment, pair[0])]
					
comment_data_total = []
comment_data_move = []
comment_data_position = []
instances = 0
instances_total = 0
for file in files:
	read_comments_of_file('files/' + file, comment_data_total, comment_data_move, comment_data_position)
end = time.time()
print(instances, instances_total, end - start)

"""
Ectract classification features
"""

print("\nEXTRACT CLASSIFICATION FEATURES\n")

def tf_features(comment, top_words, top_bigrams, top_trigrams):
	features = Counter()
	features['NUMBER_OF_TOKENS'] = len(comment)
	for word in comment:
		if word in top_words:
			features[word] += 1
	for bigram in nltk.bigrams(comment):
		if bigram in top_bigrams:
			features[bigram] += 1
	for trigram in nltk.trigrams(comment):
		if trigram in top_trigrams:
			features[trigram] += 1
	return features

def create_tf_featuresets(comment_data, top_words_total, top_bigrams_total, top_trigrams_total, top_words_move, top_bigrams_move, top_trigrams_move, top_words_position, top_bigrams_position, top_trigrams_position):
	fd_words_total = nltk.FreqDist()
	fd_bigrams_total = nltk.FreqDist()
	fd_trigrams_total = nltk.FreqDist()
	fd_words_move = nltk.FreqDist()
	fd_bigrams_move = nltk.FreqDist()
	fd_trigrams_move = nltk.FreqDist()
	fd_words_position = nltk.FreqDist()
	fd_bigrams_position = nltk.FreqDist()
	fd_trigrams_position = nltk.FreqDist()
	
	global instances
	for (words, label) in comment_data:
		instances += 1
		if not (instances % 1000):
			end = time.time()
			print(instances, end - start)
		if STEMMING:
			words = [stemmer.stem(word) for word in words]
		if REMOVE_STOPWORDS:
			words = set(words) - set(stopwords.words('english'))
		for word in words:
			fd_words_total[word] += 1
			if output_class_move_1[label]:
				fd_words_move[word] += 1
			if output_class_position_1[label]:
				fd_words_position[word] += 1
		if len(words) > 1:
			for bigram in nltk.bigrams(words):
				fd_bigrams_total[bigram] += 1
				if output_class_move_1[label]:
					fd_bigrams_move[bigram] += 1
				if output_class_position_1[label]:
					fd_bigrams_position[bigram] += 1
			for trigram in nltk.trigrams(words):
				fd_trigrams_total[trigram] += 1
				if output_class_move_1[label]:
					fd_trigrams_move[trigram] += 1
				if output_class_position_1[label]:
					fd_trigrams_position[trigram] += 1
	top_words_total += [word for (word, freq) in fd_words_total.most_common() if freq > MIN_FREQ_WORD]
	top_bigrams_total += [bigram for (bigram, freq) in fd_bigrams_total.most_common() if freq > MIN_FREQ_BIGRAM]
	top_trigrams_total += [trigram for (trigram, freq) in fd_trigrams_total.most_common() if freq > MIN_FREQ_TRIGRAM]
	top_words_move += [word for (word, freq) in fd_words_move.most_common() if freq > MIN_FREQ_WORD]
	top_bigrams_move += [bigram for (bigram, freq) in fd_bigrams_move.most_common() if freq > MIN_FREQ_BIGRAM]
	top_trigrams_move += [trigram for (trigram, freq) in fd_trigrams_move.most_common() if freq > MIN_FREQ_TRIGRAM]
	top_words_position += [word for (word, freq) in fd_words_position.most_common() if freq > MIN_FREQ_WORD]
	top_bigrams_position += [bigram for (bigram, freq) in fd_bigrams_position.most_common() if freq > MIN_FREQ_BIGRAM]
	top_trigrams_position += [trigram for (trigram, freq) in fd_trigrams_position.most_common() if freq > MIN_FREQ_TRIGRAM]
	print("Top words/bigrams/trigrams total:")
	print(len(top_words_total), len(top_bigrams_total), len(top_trigrams_total))
	print("Top words/bigrams/trigrams move:")
	print(len(top_words_move), len(top_bigrams_move), len(top_trigrams_move))
	print("Top words/bigrams/trigrams position:")
	print(len(top_words_position), len(top_bigrams_position), len(top_trigrams_position))
	featuresets = [(tf_features(comment, top_words_total, top_bigrams_total, top_trigrams_total), label) for (comment, label) in comment_data]
	return featuresets
	
top_words_total = []
top_bigrams_total = []
top_trigrams_total = []
top_words_move = []
top_bigrams_move = []
top_trigrams_move = []
top_words_position = []
top_bigrams_position = []
top_trigrams_position = []
instances = 0
featuresets = create_tf_featuresets(comment_data_total, top_words_total, top_bigrams_total, top_trigrams_total, top_words_move, top_bigrams_move, top_trigrams_move, top_words_position, top_bigrams_position, top_trigrams_position)

end = time.time()
print(end - start)

"""
Write arff files
"""

print("\nWRITE ARFF FILES\n")

def tf_write_arff(problem_name, classes, output_class, top_words, top_bigrams, top_trigrams):
    arff = open("chess-annotations-tf-" + problem_name + ".arff", "w")
    RELATION_NAME = "comment-" + problem_name									 
    buff = ("@RELATION " + RELATION_NAME + "\n")
    buff += ("@ATTRIBUTE NUMBER_OF_TOKENS INTEGER\n") 
    for word in top_words:
        buff += ("@ATTRIBUTE COUNT(" + str(word).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_") + ") INTEGER\n")
    for bigram in top_bigrams:
        buff += ("@ATTRIBUTE COUNT(" + str(bigram[0]).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_")
		+ "_" + str(bigram[1]).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_") + ") INTEGER\n")
    for trigram in top_trigrams:
        buff += ("@ATTRIBUTE COUNT(" + str(trigram[0]).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_")
		 + "_" + str(trigram[1]).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_")
		  + "_" + str(trigram[2]).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_") + ") INTEGER\n")
    buff += ("@ATTRIBUTE CLASS {" + classes + "}\n") 
    buff += ("@DATA\n")
    for (features, label) in featuresets:
        buff += ("{0 " + str(features['NUMBER_OF_TOKENS']) + ", ")        
        for idx, f in enumerate(top_words):
            if features[f] > 0:
                buff += (str(idx + 1) + " " + str(features[f]) + ", ")
        for idx, f in enumerate(top_bigrams):
            if features[f] > 0:
                buff += (str(len(top_words) + idx + 1) + " " + str(features[f]) + ", ")
        for idx, f in enumerate(top_trigrams):
            if features[f] > 0:
                buff += (str(len(top_words) + len(top_bigrams) + idx + 1) + " " + str(features[f]) + ", ")
        buff += (str(len(top_words) + len(top_bigrams) + len(top_trigrams) + 1) + " " + str(output_class[label]) + "}\n")        
    arff.write(buff)

tf_write_arff("total", "1, 2", output_class_total, top_words_total, top_bigrams_total, top_trigrams_total)  
tf_write_arff("move-1", "1, 2", output_class_move_1, top_words_move, top_bigrams_move, top_trigrams_move)  
tf_write_arff("move-2", "1, 2, 3, 4, 5, 6", output_class_move_2, top_words_move, top_bigrams_move, top_trigrams_move)  
tf_write_arff("position-1", "1, 2, 3", output_class_position_1, top_words_position, top_bigrams_position, top_trigrams_position)  
tf_write_arff("position-2", "1, 2, 3, 4, 5, 6, 7", output_class_position_2, top_words_position, top_bigrams_position, top_trigrams_position)  

end = time.time()
print(end - start)
