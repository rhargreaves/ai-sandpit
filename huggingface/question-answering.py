from pprint import pprint
from transformers import pipeline

question_answerer = pipeline("question-answering")
while True:
    context = input("Enter context: ")
    question = input("Enter question: ")
    answer = question_answerer(question, context)
    pprint(answer)

# Interesting answers

"""
Context: My husky dog has 2 ears. But he cannot hear out of one of them sadly
Question: Is my dog deaf?
{'answer': 'he cannot hear out of one of them',
 'end': 62,
 'score': 0.15252797305583954,
 'start': 29}

Context: My husky dog has 2 ears. But he cannot hear out of one of them sadly
Question: What breed is my dog?
{'answer': 'husky', 'end': 8, 'score': 0.9013784527778625, 'start': 3}

Context: My husky dog has 2 ears. But he cannot hear out of one of them sadly.
    This means he struggles to hear things coming up to him from the left
Question: Which of his ears can he not hear out of?
{'answer': 'one', 'end': 54, 'score': 0.3853362798690796, 'start': 51}
"""
