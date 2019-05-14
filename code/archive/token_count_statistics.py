import nltk
import re

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
	'chessbasedb1.pgn',
	'chessbasedb2.pgn',
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

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

tokenizer = RegexpTokenizer('#[\w\d]{2}|\$\d+|[!\?]+|[\-\+/=]+|1/2|(?:\w\.)+|\.+|[\w\d\-\']+|\S')
lemmatizer = WordNetLemmatizer()
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

"""
Extract comments of .pgn-file
"""

def read_comments_of_file(file, comment_data):
	raw = open(file, 'rb').read()
	str = raw.decode('iso-8859-1')
	
	pairs = re.findall(r'\$(?P<class>[0-9]+)\s*\{(?P<comment>[^{}]*)\}', str)
	pairs += re.findall(r'\$(?P<class>[0-9]+)\s*\$[0-9]+\s*\{(?P<comment>[^{}]*)\}', str)
	pairs += re.findall(r'(?P<class>[!\?]{1,2})\s*\{(?P<comment>[^{}]*)\}', str)
	comment_data += [(tokenizer.tokenize(pair[1].lower()), dict_total[pair[0]]) for pair in pairs]
	
comment_data = []
for file in files:
	read_comments_of_file('files/pgn/' + file, comment_data)

from nltk.corpus import stopwords

fdist_token_count = nltk.FreqDist()
for comment in comment_data:
	fdist_token_count[len(comment[0])] += 1

print("\nComments by token count\n")
fdist_token_count.tabulate()

fdist_tokens = nltk.FreqDist()
fdist_words = nltk.FreqDist()
fdist_no_stopwords = nltk.FreqDist()
fdist_bigrams = nltk.FreqDist()
fdist_trigrams = nltk.FreqDist()
fdist_word_bigrams = nltk.FreqDist()
fdist_word_trigrams = nltk.FreqDist()

for comment in comment_data:
	for token in comment[0]:
		if (lemmatizer.lemmatize(token, 'n') in english_vocab or lemmatizer.lemmatize(token, 'v') in english_vocab):
			if token not in stopwords.words("english"):
				fdist_no_stopwords[token] += 1
			fdist_words[token] += 1
		fdist_tokens[token] += 1
	if len(comment[0]) > 1:
		for bigram in nltk.bigrams(comment[0]):
			if (lemmatizer.lemmatize(bigram[0], 'n') in english_vocab or lemmatizer.lemmatize(bigram[0], 'v') in english_vocab) and (lemmatizer.lemmatize(bigram[1], 'n') in english_vocab or lemmatizer.lemmatize(bigram[1], 'v') in english_vocab):
				fdist_word_bigrams[bigram[0] + " " + bigram[1]] += 1
			fdist_bigrams[bigram[0] + " " + bigram[1]] += 1
	if len(comment[0]) > 2:
		for trigram in nltk.trigrams(comment[0]):
			if (lemmatizer.lemmatize(trigram[0], 'n') in english_vocab or lemmatizer.lemmatize(trigram[0], 'v') in english_vocab) and (lemmatizer.lemmatize(trigram[1], 'n') in english_vocab or lemmatizer.lemmatize(trigram[1], 'v') in english_vocab) and (lemmatizer.lemmatize(trigram[2], 'n') in english_vocab or lemmatizer.lemmatize(trigram[2], 'v') in english_vocab):
				fdist_word_trigrams[trigram[0] + " " + trigram[1] + " " + trigram[2]] += 1
			fdist_trigrams[trigram[0] + " " + trigram[1] + " " + trigram[2]] += 1

print("\nTokens by count\n")
fdist_tokens.tabulate(30)
print("\nWords by count\n")
fdist_words.tabulate(30)
print("\nWords by count without stopwords\n")
fdist_no_stopwords.tabulate(30)
print("\nBigrams by count\n")
fdist_bigrams.tabulate(30)
print("\nTrigrams by count\n")
fdist_trigrams.tabulate(30)
print("\nWord bigrams by count\n")
fdist_word_bigrams.tabulate(30)
print("\nWord trigrams by count\n")
fdist_word_trigrams.tabulate(30)

print("\nGrouped count tokens\n")
for limit in {1000, 300, 100, 30, 10, 3, 1}:
	print(limit, len(list(filter(lambda x: x[1] >= limit, fdist_tokens.items()))))
print("\nGrouped count bigrams\n")
for limit in {1000, 300, 100, 30, 10, 3, 1}:
	print(limit, len(list(filter(lambda x: x[1] >= limit, fdist_bigrams.items()))))
print("\nGrouped count trigrams\n")
for limit in {1000, 300, 100, 30, 10, 3, 1}:
	print(limit, len(list(filter(lambda x: x[1] >= limit, fdist_trigrams.items()))))