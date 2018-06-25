import json

''' This file loads the data and randomizes the order or the sentences'''

def getTopics(devSet):
	TRAINFILE = "../data/tuples-train.json"
	OUTPUT_FILE = "../data/tuples-output-train.json"
	if devSet:
		TRAINFILE = "../data/tuples-dev.json"
		OUTPUT_FILE = "../data/tuples-output-dev.json"
	print("Extracting...", "\n")
	with open(TRAINFILE, encoding = "utf8") as f:
		json_set = json.load(f)
		dataset = json_set['data']
		print(json_set['version'])

def pairs(devSet):
	pairList = []
	data = getTopics(devSet)
	for topic in data:
		for paragraph in data['paragraphs']:
			for pair in paragraph['pairs']:
				pairList.append(pair)
	return pairList