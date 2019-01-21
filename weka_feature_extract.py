import nltk
import re

"""
Extract comments of .pgn-file
"""

files = [
	'bali02.pgn',
	'chessdoctor.pgn',
	'd00_chess_informant.pgn',
	'electronic_campfire.pgn',
	'exeter_lessons_from_tal.pgn',
	'famous_games.pgn',
	'GM_games.pgn',
	'great_masters.pgn',	
	'hartwig.pgn',
	'hayes.pgn',
	'human_computer.pgn',
	'immortal_games.pgn',
	'kk.pgn',
	'koltanowski.pgn',
	'kramnik.pgn',
	'linares_2001.pgn',
	'linares_2002.pgn',
	'middleg.pgn',
	'moscow64.pgn',
	'newyork1924.pgn',
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
]

dict = {
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
	"138": 0,
	"142": 0,
	"146": 0,
}

from nltk.tokenize import RegexpTokenizer

def read_comments_of_file(file, comment_data):
	raw = open(file, 'rb').read()
	str = raw.decode('iso-8859-1')
	tokenizer = RegexpTokenizer('(?:\w\.)+|[\w\d\-\']+|\.+|#[\w\d]{2}|1/2|[!\?]+|[\-\+/=]+|\S')
    
	comments = re.findall(r'\$(?P<class>[0-9]+)\s*{(?P<comment>[^{}]*)}', str)		#NAG
	comments += re.findall(r'(?P<class>[!\?]{1,2})\s*{(?P<comment>[^{}]*)}', str)	#symbol

	comment_data += [(tokenizer.tokenize(pair[1].lower()), dict[pair[0]]) for pair in comments if dict[pair[0]]]

comment_data = []
for file in files:
	read_comments_of_file('files/' + file, comment_data)	

"""
Ectract classification features
"""

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def rating_features(comment):
	features = {}
	for word in top_words:
		features[word] = comment.count(word)
	for bigram in top_bigrams:
		features[bigram] = comment.count(bigram)
	for trigram in top_trigrams:
		features[trigram] = comment.count(trigram)
	return features

#threshold = 1000
fd_all_words = nltk.FreqDist()
fd_all_bigrams = nltk.FreqDist()
fd_all_trigrams = nltk.FreqDist()
stemmer = PorterStemmer()
for (words,_) in comment_data:
	#print(words)
	#words = [stemmer.stem(word) for word in words]
	for w in words: #set(words) - set(stopwords.words('english')):
		fd_all_words[w] += 1
	if len(words) > 1:
		for b in nltk.bigrams(words):
			fd_all_bigrams[b] += 1
		for t in nltk.trigrams(words):
			fd_all_trigrams[t] += 1
top_words = [word for (word, freq) in fd_all_words.most_common() if freq > 25]
top_bigrams = [bigram for (bigram, freq) in fd_all_bigrams.most_common() if freq > 500]
top_trigrams = [trigram for (trigram, freq) in fd_all_trigrams.most_common() if freq > 50]
print(len(top_words))
print(len(top_bigrams))
print(len(top_trigrams))
featuresets = [(rating_features(c), g) for (c,g) in comment_data]

"""
Write arff file
"""

def write_arff():
    arff = open("chess-annotations.arff", "w")
    RELATION_NAME = "comment"									 
    arff.write("@RELATION " + RELATION_NAME + "\n")
    for word in top_words:
        arff.write("@ATTRIBUTE COUNT(" + str(word).replace("\"", "_quote_").replace("'", "_apostrophe_").replace(",", "_comma_").replace("%", "_percent_") + ") INTEGER\n")
    for bigram in top_bigrams:
        arff.write("@ATTRIBUTE COUNT(" + str(bigram).replace("\"", "_quote_").replace(" ", "").replace("'", "").replace(",", "_").replace("%", "_percent_") + ") INTEGER\n")
    for trigram in top_trigrams:
        arff.write("@ATTRIBUTE COUNT(" + str(trigram).replace("\"", "_quote_").replace(" ", "").replace("'", "").replace(",", "_").replace("%", "_percent_") + ") INTEGER\n")
    arff.write("@ATTRIBUTE CLASS {1, 2, 3, 4, 5, 6}\n") 
    arff.write("@DATA\n")
    for (features, g) in featuresets:
        buff = ""
        buff += (",".join([str(features[f]) for f in features]) + ", " + str(g) + "\n")        
        arff.write(buff)

write_arff()  
