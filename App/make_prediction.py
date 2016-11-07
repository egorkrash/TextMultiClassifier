from predictor import predict_speaker
import sys

try:
	text = sys.argv[1]
except IndexError:
	f = open('input.txt')
	text = f.read()
	text = text.strip()
	f.close()

if len(text) < 50:
	print 'we need more text to make better prediction :(. try again'
else: 
	print predict_speaker(text)