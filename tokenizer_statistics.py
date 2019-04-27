import nltk
import re

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
	'chessbasedb.pgn',
]

dict_total = {
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

dict_move_1 = {
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

dict_move_2 = {
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

dict_position_1 = {
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

dict_position_2 = {
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

tokenizer = RegexpTokenizer('#[\w\d]{2}|\$\d+|[!\?]+|[\-\+/=]+|1/2|(?:\w\.)+|\.+|[\w\d\-\']+|\S')
lemmatizer = WordNetLemmatizer()
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

"""
Extract comments of .pgn-file
"""

def read_comments_of_file(file, comment_data_total, comment_data_move_1, comment_data_move_2, comment_data_position_1, comment_data_position_2):
	raw = open(file, 'rb').read()
	str = raw.decode('iso-8859-1')
	
	pairs = re.findall(r'\$(?P<class>[0-9]+)\s*\{(?P<comment>[^{}]*)\}', str)		#NAG
	pairs += re.findall(r'\$(?P<class>[0-9]+)\s*\$[0-9]+\s*\{(?P<comment>[^{}]*)\}', str)		#NAG
	pairs += re.findall(r'(?P<class>[!\?]{1,2})\s*\{(?P<comment>[^{}]*)\}', str)	#symbol
	
	#buff = ""
	for pair in pairs:
		test = identifier.classify(pair[1])
		if test[0] != "en" and test[1] > 0.99:
			#buff += ("N1 " + '{:>3}'.format(pair[0]) + " " + pair[1].replace("\n", " ").replace("\r", " ") + "\n")
			continue
		if dict_total[pair[0]]:
			comment = tokenizer.tokenize(pair[1].lower())
			count = 0
			for token in comment:
				if (lemmatizer.lemmatize(token, 'n') in english_vocab or lemmatizer.lemmatize(token, 'v') in english_vocab) and count == 3:
					break
				if (lemmatizer.lemmatize(token, 'n') in english_vocab or lemmatizer.lemmatize(token, 'v') in english_vocab) and count < 3:
					count += 1
			if count == 3:
				#buff += ("Y  ")		    
				comment_data_total += [(comment, dict_total[pair[0]])]
				if dict_move_1[pair[0]]:
					comment_data_move_1 += [(comment, dict_move_1[pair[0]])]
					comment_data_move_2 += [(comment, dict_move_2[pair[0]])]
				if dict_position_1[pair[0]]:
					comment_data_position_1 += [(comment, dict_position_1[pair[0]])]
					comment_data_position_2 += [(comment, dict_position_2[pair[0]])]
	"""
			if count < 3:
				buff += ("N3 ")
		else:
			buff += ("N2 ")   
		buff += ('{:>3}'.format(pair[0]) + " " + pair[1].replace("\n", " ").replace("\r", " ") + "\n")
	instances.write(buff)
	"""
	
	"""
	fdist_complete = nltk.FreqDist()
	fdist_move = nltk.FreqDist()
	fdist_position = nltk.FreqDist()
	for pair in pairs:
		fdist_complete[pair[0]] += 1
		fdist_move[dict_move_1[pair[0]]] += 1
		fdist_position[dict_position_1[pair[0]]] += 1
	cfd_complete[file] = fdist_complete
	cfd_move[file] = fdist_move
	cfd_position[file] = fdist_position
	"""

"""
cfd_total = nltk.ConditionalFreqDist()	
cfd_move = nltk.ConditionalFreqDist()	
cfd_position = nltk.ConditionalFreqDist()	
"""

#instances = open("instances.txt", "w")
comment_data_total = []
comment_data_move_1 = []
comment_data_move_2 = []
comment_data_position_1 = []
comment_data_position_2 = []
for file in files:
	read_comments_of_file('files/' + file, comment_data_total, comment_data_move_1, comment_data_move_2, comment_data_position_1, comment_data_position_2)

"""
cfd_total.tabulate()	
cfd_move.tabulate()	
cfd_position.tabulate()
"""

print("\nComments by token count\n")
fdist_token_count = nltk.FreqDist()
for comment in comment_data_total:
	fdist_token_count[len(comment[0])] += 1
#fdist_token_count.tabulate()

print("\nTokens by count\n")
fdist_tokens = nltk.FreqDist()
for comment in comment_data_total:
	for token in comment[0]:
		fdist_tokens[token] += 1
#fdist_tokens.tabulate()

"""
Ectract classification features
"""

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def rating_features(comment, top_words, top_bigrams, top_trigrams):
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

def create_featuresets(comment_data, top_words, top_bigrams, top_trigrams):
	#threshold = 1000
	fd_all_words = nltk.FreqDist()
	fd_all_bigrams = nltk.FreqDist()
	fd_all_trigrams = nltk.FreqDist()
	stemmer = PorterStemmer()
	for (words,_) in comment_data:
		#print(words)
		if STEMMING:
			words = [stemmer.stem(word) for word in words]
		if REMOVE_STOPWORDS:
			words = set(words) - set(stopwords.words('english'))
		for w in words:
			fd_all_words[w] += 1
		if len(words) > 1:
			for b in nltk.bigrams(words):
				fd_all_bigrams[b] += 1
			for t in nltk.trigrams(words):
				fd_all_trigrams[t] += 1
	top_words += [word for (word, freq) in fd_all_words.most_common() if freq > MIN_FREQ_WORD]
	top_bigrams += [bigram for (bigram, freq) in fd_all_bigrams.most_common() if freq > MIN_FREQ_BIGRAM]
	top_trigrams += [trigram for (trigram, freq) in fd_all_trigrams.most_common() if freq > MIN_FREQ_TRIGRAM]
	#print(len(top_words))
	#print(len(top_bigrams))
	#print(len(top_trigrams))
	featuresets = [(rating_features(c, top_words, top_bigrams, top_trigrams), g) for (c,g) in comment_data]
	return featuresets
	
top_words_total = []
top_bigrams_total = []
top_trigrams_total = []
top_words_move_1 = []
top_bigrams_move_1 = []
top_trigrams_move_1 = []
top_words_move_2 = []
top_bigrams_move_2 = []
top_trigrams_move_2 = []
top_words_position_1 = []
top_bigrams_position_1 = []
top_trigrams_position_1 = []
top_words_position_2 = []
top_bigrams_position_2 = []
top_trigrams_position_2 = []
featuresets_total = create_featuresets(comment_data_total, top_words_total, top_bigrams_total, top_trigrams_total)
featuresets_move_1 = create_featuresets(comment_data_move_1, top_words_move_1, top_bigrams_move_1, top_trigrams_move_1)
featuresets_move_2 = create_featuresets(comment_data_move_2, top_words_move_2, top_bigrams_move_2, top_trigrams_move_2)
featuresets_position_1 = create_featuresets(comment_data_position_1, top_words_position_1, top_bigrams_position_1, top_trigrams_position_1)
featuresets_position_2 = create_featuresets(comment_data_position_2, top_words_position_2, top_bigrams_position_2, top_trigrams_position_2)

"""
Write arff file
"""

def write_arff(featuresets, problem_name, classes, top_words, top_bigrams, top_trigrams):
    arff = open("chess-annotations-" + problem_name + ".arff", "w")
    RELATION_NAME = "comment"									 
    buff = ("@RELATION " + RELATION_NAME + "\n")
    for word in top_words:
        buff += ("@ATTRIBUTE COUNT(" + str(word).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_") + ") INTEGER\n")
    for bigram in top_bigrams:
        buff += ("@ATTRIBUTE COUNT(" + str(bigram[0]).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_")
		+ "_" + str(bigram[1]).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_") + ") INTEGER\n")
    for trigram in top_trigrams:
        buff += ("@ATTRIBUTE COUNT(" + str(trigram[0]).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_")
		 + "_" + str(trigram[1]).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_")
		  + "_" + str(trigram[2]).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_") + ") INTEGER\n")
    buff += ("@ATTRIBUTE NUMBER_OF_TOKENS INTEGER\n") 
    buff += ("@ATTRIBUTE CLASS {" + classes + "}\n") 
    buff += ("@DATA\n")
    for (features, g) in featuresets:
        buff += ("{")
        for idx, f in enumerate(top_words):
            if features[f] > 0:
                buff += (str(idx) + " " + str(features[f]) + ", ")
        for idx, f in enumerate(top_bigrams):
            if features[f] > 0:
                buff += (str(len(top_words) + idx) + " " + str(features[f]) + ", ")
        for idx, f in enumerate(top_trigrams):
            if features[f] > 0:
                buff += (str(len(top_words) + len(top_bigrams) + idx) + " " + str(features[f]) + ", ")
        buff += (str(len(top_words) + len(top_bigrams) + len(top_trigrams)) + " " + str(features['NUMBER_OF_TOKENS']) + ", ")        
        buff += (str(len(top_words) + len(top_bigrams) + len(top_trigrams) + 1) + " " + str(g) + "}\n")        
    arff.write(buff)

write_arff(featuresets_total, "total", "1, 2", top_words_total, top_bigrams_total, top_trigrams_total)  
write_arff(featuresets_move_1, "move-1", "1, 2", top_words_move_1, top_bigrams_move_1, top_trigrams_move_1)  
write_arff(featuresets_move_2, "move-2", "1, 2, 3, 4, 5, 6", top_words_move_2, top_bigrams_move_2, top_trigrams_move_2)  
write_arff(featuresets_position_1, "position-1", "1, 2, 3", top_words_position_1, top_bigrams_position_1, top_trigrams_position_1)  
write_arff(featuresets_position_2, "position-2", "1, 2, 3, 4, 5, 6, 7", top_words_position_2, top_bigrams_position_2, top_trigrams_position_2)  