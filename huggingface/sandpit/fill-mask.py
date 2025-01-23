from pprint import pprint
from transformers import pipeline

unmasker = pipeline("fill-mask", "distilbert/distilroberta-base")
while True:
    text = input("Enter text with <mask>: ")
    result = unmasker(text, top_k=2)
    pprint(result)

# the sun today is <mask>
# -----------------------
# [{'score': 0.13146445155143738,
#   'sequence': 'the sun today is shining',
#   'token': 21003,
#  'token_str': ' shining'},
# {'score': 0.03517243638634682,
#  'sequence': 'the sun today is bright',
#  'token': 4520,
#  'token_str': ' bright'}]
