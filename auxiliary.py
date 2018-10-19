'''
Utility functions for Entety Tagger Excersice

Author: Jesus I. Ramirez Franco
October 2018
'''

import nltk
from nltk.tag import StanfordNERTagger
import os
import re
from nltk.corpus import stopwords
import pandas as pd

model_path = 'C:/Users/jesus/OneDrive/Documentos/GitHub/NLP_test/stanford-ner-2018-02-27/classifiers/english.muc.7class.distsim.crf.ser.gz'
jar_path = 'C:/Users/jesus/OneDrive/Documentos/GitHub/NLP_test/stanford-ner-2018-02-27/stanford-ner.jar'
java_path = 'C:/Program Files/Java/jdk-11/bin/java.exe'

os.environ['JAVAHOME'] = java_path

# Tagger object
nert = StanfordNERTagger(model_path, jar_path)

def files_to_raw(file_names):
	'''
	Open txt files and creates a dictionary with the name of
	the file as key and raw text as value.
	Inputs:
		file_names(list): list of paths to files to be analyzed.
	Returns: a dictionary.
	'''
	text_dict = {}
	for name in file_names:
		f = open(name, encoding = 'ANSI')
		text_dict[name] = f.read()

	return text_dict
		

def tagger(text, tag, tagger_=nert):
	'''
	Tags the words of a text according to Stanford_NER classifier, and tracks
	only the words with the desired tags; for instance, 'MONEY'.
	Inputs:
		text (string): Raw text to be analyzed.
		tag (string): tag of interest.
		tagger_: model of classifier used.
	Returns: list of classified words and the indices where they locate in
	the tokenized text.
	'''
	tokenized_text = nltk.word_tokenize(text)
	tagged_words = tagger_.tag(tokenized_text)
	words = []
	indices = []
	
	for i in range(len(tagged_words)):
		if tagged_words[i][1]==tag:
			words.append(tagged_words[i][0])
			indices.append(i)
	
	return words, indices 


def token_intervals(sentences): 
	'''
	Identifies the intervals of indices in the word-tokenized text 
	that correspond to the indices in the sentence-tokenized text.
	Inputs:
		sentences(list): list of strings with the text divided in
		sentences.
	Returns: a dictionary with the interval of indices of words 
	that contanis each sentence.  
	'''
	sentences_len = {}
	for i in range(len(sentences)):
		sentences_len[i] = len(nltk.word_tokenize(sentences[i]))

	intervals = {}
	low = 0
	high = 0
	for k, v in sentences_len.items():
		high += v
		intervals[(low, high)] = k
		low += v

	return intervals

def find_sentences(text, tag='MONEY'):
	'''
	Identifies the sentences that contain at least a word classified 
	with the tag of interest; for instance 'MONEY'.
	Inputs:
		text(string): raw text to be analized.
	Returns: a list of sentences.
	'''
	sentences = nltk.sent_tokenize(text)
	words, indices = tagger(text, tag)
	intervals = token_intervals(sentences)

	tagged_sentences = []
	for i in indices:
		for k, v in intervals.items():
			if i in range(k[0], k[1]):
				if sentences[v] not in tagged_sentences:
					tagged_sentences.append(sentences[v])

	return tagged_sentences

def get_entity_sentences(file_names, tag='MONEY'):
	'''
	Analyze a text and retuns the sentences of that text that have at least
	one word classified with the tag of interest; for instance 'MONEY'.
	Inputs:
		file_names(list): list of paths to files to be analyzed.
	Returns: a dictinary with the name of the file as key and a list of
	sentences as value.
	'''
	files_dict = files_to_raw(file_names)
	results_dict = {}

	for k, v in files_dict.items():
		results_dict[k] = find_sentences(v)

	return results_dict

'''
The following additional auxiliary functions help to clean and show the results.
The cleaning process includes removing unknown characters, change capital 
to lower letters and remove english stop words.
'''


def show_sentences(results, file_name):
	'''
	Prints the sentences found in results
	Inputs:
		results (dictionary): dictionary with results.
		file_name (string): name of one of the keys in results.
	Results: Nothing, just prints the results.
	'''
	if file_name not in results.keys():
		print('Not a valid file name')
	else:
		print('Sentences found in', file_name)
		print('-----------------------------------------------------------------')
		for sentence in results[file_name]:
			print(sentence)
			print('*************************************************************')

def clean_sentence(sentence):
	'''
	Removes unknown characters, change capital to lower letters and remove
	english stop words
	Inputs:
		sentence (string): a sting to be cleaned
	Returns: a string
	'''
	new = ''
	for l in sentence:
		if re.match('[a-zA-Z0-9_\s]',l):
			new += l

	tokens = nltk.word_tokenize(new)
	tokens = [t.lower() for t in tokens]

	new_tokens = []
	for t in tokens:
		if t not in set(stopwords.words('english')):
			new_tokens.append(t)

	return ' '.join(new_tokens)

def clean_results(results):
	'''
	Createsa new results dictionary with cleaned sentences
	Inputs:
		results (dictionary): Dictionary with results to be cleaned.
	Returns: a cleaned dictionary.
	'''
	new_dict = {}
	for k, v in results.items():
		sentences = v
		new_dict[k] = [clean_sentence(s) for s in sentences]

	return new_dict

def tokens_freq(corpus, size):
	'''
	Computes the frequency of n-grams according to size and
	retuns an ordered data frame.
	Inputs:
		corpus (string): text to be analized
		size (int): size of n-grams
	Returns: a data frame
	'''
	
	tokens = nltk.word_tokenize(corpus)
	frequencies = {}
	complete = tokens + tokens[:size - 1]

	n_grams = []
	for i in range(len(tokens)):

		l = i
		h = i + size-1
		n_grams.append(''+complete[l]+','+complete[h])

	for ng in n_grams:
		if ng not in frequencies.keys():
			frequencies[ng] = 1
		else:
			frequencies[ng] += 1

	freq_list = [(k, v) for k, v in frequencies.items()]
	df = pd.DataFrame(freq_list, columns=[str(size)+'-gram', 'Frequency'])
	return df.sort_values(by='Frequency', ascending=False)[:10]



