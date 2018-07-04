import torch
import spacy
import re

nlp = spacy.load('en_core_web_sm')

class TupleCritic():
	''' Loss function that tries to make output have good tuples '''
	def forward(self, sentence, model):
		doc = nlp(sentence)
		ents = [i.lower_ for i in doc.ents]
		nouns = [word.lower_ for sent in list(doc.sents) for word in sent if word.pos_.endswith("NOUN") or word.pos_ == "NUM"]
		reward = 0
		if model[0] == "<":
			reward += 1
		if model[-1] == ">":
			reward += 1
		groups = re.search(r'<([^>]+)>', model)
		commas = 0
		for i in groups.group(0):
			if i == ",":
				commas += 1
		if commas == 2:
			reward += 1
			# Now check to see if named entities are in the head and the tail
			arr = groups.group(0).split(",")
			if arr[0].lower() in ents:
				reward += 1
			if arr[2].lower() in ents:
				reward += 1
			for i in arr[0]:
				if i.lower() in nouns:
					reward += 1
					nouns.remove(i) # We don't want it to also be in the tail if in the head
			for i in arr[2]:
				if i.lower() in nouns:
					reward += 1
		return reward
