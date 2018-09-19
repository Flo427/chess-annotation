import nltk
import random
import re

"""
Extract comments of .pgn-file
"""

files = [
	'bali02.pgn',
	'chessdoctor.pgn',
	'd00_chess_informant.pgn',
	'electronic_campfire.pgn',
#	'europe_echecs.pgn',
	'exeter_lessons_from_tal.pgn',
	'famous_games.pgn',
	'GM_games.pgn',
	'great_masters.pgn',	
	'hartwig.pgn',
	'hayes.pgn',
	'human_computer.pgn',
	'immortal_games.pgn',
#	'kasp_top.pgn',
	'kk.pgn',
	'koltanowski.pgn',
	'kramnik.pgn',
	'linares_2001.pgn',
	'linares_2002.pgn',
	'middleg.pgn',
	'moscow64.pgn',
	'newyork1924.pgn',
#	'perle.pgn',
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
	"1": 1,
	"2": 2,
	"3": 3,
	"4": 4,
	"5": 5,
	"6": 6,
	"!": 1,
	"?": 2,
	"!!": 3,
	"??": 4,
	"!?": 5,
	"?!": 6,
	"7": 0,
	"8": 0,
	"10": 0,
	"11": 11,
	"12": 11,
	"13": 13,
	"14": 14,
	"15": 15,
	"16": 14,
	"17": 15,
	"18": 14,
	"19": 15,
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

comment_data = []

def read_comments_of_file(file, comment_data, cfd):
	raw = open(file, 'rb').read()
	str = raw.decode('iso-8859-1')

	comments = re.findall(r'\$(?P<class>[0-9]+)\s*{(?P<comment>[^{}]*)}', str)		#NAG
	comments += re.findall(r'(?P<class>[!\?]{1,2})\s*{(?P<comment>[^{}]*)}', str)	#symbol
		
	comment_data += [(pair[1].lower().split(), dict[pair[0]]) for pair in comments if dict[pair[0]]]

	fdist = nltk.FreqDist()
	for pair in comments:
		fdist[dict[pair[0]]] += 1
	cfd[file] = fdist

cfd = nltk.ConditionalFreqDist()	
for file in files:
	read_comments_of_file('files/' + file, comment_data, cfd)	
#cfd.tabulate()
	
random.shuffle(comment_data)

"""
Ectract classification features
"""

threshold = 1000
fd_all_words = nltk.FreqDist()
for (words,_) in comment_data:
	for w in words:
		fd_all_words[w] += 1
top_words = [word for (word, freq) in fd_all_words.most_common(threshold)]

def rating_features(comment):
	features = {}
	for word in top_words:
		features[word] = comment.count(word)
	return features

"""
Prepare data
"""

count = len(comment_data)

cut1 = int(0.7 * count)
cut2 = int(0.9 * count)

featuresets = [(rating_features(c), g) for (c,g) in comment_data]
training_data, devtest_data, test_data = comment_data[:cut1], comment_data[cut1:cut2], comment_data[cut2:]
training_fdata, devtest_fdata, test_fdata = featuresets[:cut1], featuresets[cut1:cut2], featuresets[cut2:]

"""
Train a classifier
"""

classifier = nltk.NaiveBayesClassifier.train(training_fdata)

"""
Error analysis
"""

errors = []
tags = []
guesses = []
for (comment, tag) in test_data:
	guess = classifier.classify(rating_features(comment))
	tags.append(tag)
	guesses.append(guess)
	if guess != tag:
		errors.append( (tag, guess, comment) )
		
#for (tag, guess, comment) in sorted(errors):
#	print('correct=%-8s guess=%-8s comment=%-100s' % (tag, guess, comment))
	
print(nltk.classify.accuracy(classifier, test_fdata))
print(classifier.show_most_informative_features(5))

print(nltk.ConfusionMatrix(tags,guesses).pretty_format())