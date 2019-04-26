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
	#'chessbasedb.pgn',
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

dict_move = {
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

dict_position = {
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
from polyglot.detect import Detector

def read_comments_of_file(file, comment_data_total, comment_data_move, comment_data_position):
	raw = open(file, 'rb').read()
	str = raw.decode('iso-8859-1')
	tokenizer = RegexpTokenizer('#[\w\d]{2}|\$\d+|[!\?]+|[\-\+/=]+|1/2|(?:\w\.)+|\.+|[\w\d\-\']+|\S')
    
	comments = re.findall(r'\$(?P<class>[0-9]+)\s*\{(?P<comment>[^{}]*)\}', str)		#NAG
	comments += re.findall(r'\$(?P<class>[0-9]+)\s*\$[0-9]+\s*\{(?P<comment>[^{}]*)\}', str)		#NAG
	comments += re.findall(r'(?P<class>[!\?]{1,2})\s*\{(?P<comment>[^{}]*)\}', str)	#symbol
	
#	for pair in comments:
#		print(pair[0])
#		print(pair[1])
#		print(Detector(pair[1], quiet=True).languages)
		
	comments_total = []
	comments_move = []
	comments_position = []
	comments_total += [(tokenizer.tokenize(pair[1].lower()), dict_total[pair[0]]) for pair in comments if dict_total[pair[0]]]
	comments_move += [(tokenizer.tokenize(pair[1].lower()), dict_move[pair[0]]) for pair in comments if dict_move[pair[0]]]
	comments_position += [(tokenizer.tokenize(pair[1].lower()), dict_position[pair[0]]) for pair in comments if dict_position[pair[0]]]
	comment_data_total += comments_total
	comment_data_move += comments_move
	comment_data_position += comments_position
	if len(comments):
		average_length = sum(len(comment[1]) for comment in comments) / len(comments)
	else:
		average_length = 0
	if len(comments):
		average_tokens = sum(len(comment[0]) for comment in comments) / len(comments)
	else:
		average_tokens = 0
	#print(len(comments), average_length, average_tokens)

	fdist_complete = nltk.FreqDist()
	fdist_move = nltk.FreqDist()
	fdist_position = nltk.FreqDist()
	for pair in comments:
		fdist_complete[pair[0]] += 1
		fdist_move[dict_move[pair[0]]] += 1
		fdist_position[dict_position[pair[0]]] += 1
	cfd_complete[file] = fdist_complete
	cfd_move[file] = fdist_move
	cfd_position[file] = fdist_position

cfd_complete = nltk.ConditionalFreqDist()	
cfd_move = nltk.ConditionalFreqDist()	
cfd_position = nltk.ConditionalFreqDist()	
comment_data_total = []
comment_data_move = []
comment_data_position = []
for file in files:
	read_comments_of_file('files/' + file, comment_data_total, comment_data_move, comment_data_position)
#cfd_total.tabulate()	
#cfd_move.tabulate()	
#cfd_position.tabulate()

print("\nComments by token count\n")
fdist_token_count = nltk.FreqDist()
for comment in comment_data_total:
	fdist_token_count[len(comment[0])] += 1
fdist_token_count.tabulate()

print("\nTokens by count\n")
fdist_tokens = nltk.FreqDist()
for comment in comment_data_total:
	for token in comment[0]:
		fdist_tokens[token] += 1
fdist_tokens.tabulate(15)
