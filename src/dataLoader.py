import json
import sys
import numpy as np



TRAINFILE = "../data/tuples-train.json"
OUTPUT_FILE = "../data/tuples-output-train.json"

if len(sys.argv) > 1 and sys.argv[1] == "dev":
	TRAINFILE = "../data/tuples-dev.json"
	OUTPUT_FILE = "../data/tuples-output-dev.json"

def gimme_Data(sentence, tuple, reverse=False):
	print("Extracting...", "\n")
	with open(TRAINFILE, encoding = "utf8") as f:
		json_set = json.load(f)
		dataset = json_set['data']