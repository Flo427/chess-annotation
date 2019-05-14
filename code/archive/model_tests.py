from gensim.models import Word2Vec

model = Word2Vec.load('chess-annotations.model')

for word in ["white", "king", "pawn", "check", "win", "blunder"]:
	print("\nMost similar words to '" + word + "':\n")
	for similar_word in model.wv.most_similar(word):
		print(similar_word[0], similar_word[1])
