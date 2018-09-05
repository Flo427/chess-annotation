import nltk
import random
import re

"""
Extract comments of .pgn-file
"""

raw = open('middleg.pgn', 'rb').read()
str = raw.decode('utf-8')

comments_with_nag = re.findall(r'\$(?P<class>[0-9]+)\s*{(?P<comment>[^{}]*)}', str)

comment_data = [(pair[1].lower().split(), pair[0])
	for pair in comments_with_nag]
	
random.shuffle(comment_data)
#print(comment_data[:5])

"""
Ectract classification features
"""

threshold = 1000
fd_all_words = nltk.FreqDist((w for w in words) for (words,_) in comment_data)
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
#print(count)

cut1 = int(0.7 * count)
cut2 = int(0.9 * count)

featuresets = [(rating_features(c), g) for (c,g) in comment_data]
training_data, devtest_data, test_data = comment_data[:cut1], comment_data[cut1:cut2], comment_data[cut2:]
training_fdata, devtest_fdata, test_fdata = featuresets[:cut1], featuresets[cut1:cut2], featuresets[cut2:]
#print(training_data[:5])

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
#print(classifier.show_most_informative_features(5))

print(nltk.ConfusionMatrix(tags,guesses).pretty_format())