# we need this for Python 2.7
from __future__ import division     

import re
import string
import numpy as np
import pandas as pd
import collections
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
	#lowercase
	text = text.lower()
	#remove numbers
	text = re.sub(r'\d+','', text)
	#remove punctuation
	text = text.translate(string.maketrans("",""),string.punctuation)
	#remove whitespaces
	text = text.strip()
	#tokenization
	tokens = word_tokenize(text)
	#remove stop_words
	tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
	#stemming
	tokens = [stemmer.stem(w) for w in tokens]
	#lemmatization
	#tokens = [lemmatizer.lemmatize(w) for w in tokens]
	return ' '.join(tokens)
	

def word_frequency(text):

	# word_frequency dictionary

	word_freq = {}
	for sentence in text:
		tokens = nltk.word_tokenize(sentence)
		for token in tokens:
			if token not in word_freq.keys():
				word_freq[token] = 1
			else:
				word_freq[token] += 1
				
	return word_freq
	
	
def bagOfWords(text,word_freq):
	
	# check each word from word_freq if it's contained in sentence
	
	sentence_vectors = []
	for sentence in text:
		sentence_tokens = nltk.word_tokenize(sentence)
		sent_vec = []
		for token in word_freq:
			if token in sentence_tokens:
				sent_vec.append(1)
			else:
				sent_vec.append(0)
		sentence_vectors.append(sent_vec)

	# from list to matrix		
	sentence_vectors = np.asarray(sentence_vectors)  

	return sentence_vectors

def word_idf(text,word_freq):

	# calculating idf values

	word_idf_values = {}
	for token in word_freq:
		sent_num = 0     # number of sentences that contain token
		for sentence in text:
			if token in nltk.word_tokenize(sentence):
				sent_num += 1
		word_idf_values[token] = np.log(len(text)/(1 + sent_num))
	
	return word_idf_values

def word_tf(text,word_freq):

	# clalculating tf values
	
	word_tf_values = {}
	for token in word_freq:
		sent_tf_vector = []
		for sentence in text:
			sent_freq = 0
			for word in nltk.word_tokenize(sentence):
				if token == word:
					sent_freq += 1
			word_tf = sent_freq/len(nltk.word_tokenize(sentence))
			sent_tf_vector.append(word_tf)
		word_tf_values[token] = sent_tf_vector	
		
	return word_tf_values
	
def tfidf(text,word_freq):

	# combining word_idf and word_tf functions 
	# TF-IDF(t,d) = TF(t,d)*IDF(t)

	word_idf_values = word_idf(text,word_freq)
	word_tf_values = word_tf(text,word_freq)
	
	tfidf_values = []
	for token in word_tf_values.keys():
		tfidf_sentences = []
		for tf_sentence in word_tf_values[token]:
			tf_idf_score = tf_sentence * word_idf_values[token]
			tfidf_sentences.append(tf_idf_score)
		tfidf_values.append(tfidf_sentences)	
	
	return np.asarray(tfidf_values);	
	
text= []

# reading text line by line from .txt file

with open(r"test.txt") as file:
	lines = file.readlines()
	lines = [line.rstrip() for line in lines]
	text = filter(None,lines)


# preprocessing each sentence from text

text = [preprocess(sentence) for sentence in text]	
	
word_freq = word_frequency(text)	

print("Using Bag of Words: ------")
bow_model = bagOfWords(text,word_freq)	
print(bow_model)

print("Using TD-IDF Model: ------")
tf_idf_model = tfidf(text,word_freq)
print(tf_idf_model)
