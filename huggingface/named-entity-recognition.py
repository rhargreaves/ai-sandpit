from pprint import pprint
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
while True:
    text = input("Enter text: ")
    result = ner(text)
    pprint(result)
