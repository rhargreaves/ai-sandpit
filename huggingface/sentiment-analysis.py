from transformers import pipeline

classifier = pipeline("sentiment-analysis")
while True:
    text = input("Enter a text to perform sentiment analysis on: ")
    sentiment = classifier(text)
    print(sentiment)

# Some interesting results:

# Isn't it a nice day outside?
# [{'label': 'NEGATIVE', 'score': 0.8359090089797974}]

# What a nice day
# [{'label': 'POSITIVE', 'score': 0.9998655319213867}]

# Pigeons fly often
# [{'label': 'POSITIVE', 'score': 0.9531066417694092}]

# Pigeons fly often in my face
# [{'label': 'NEGATIVE', 'score': 0.9815180897712708}

# Pigeons fly often in my face displaying their rich plumage
# [{'label': 'POSITIVE', 'score': 0.9982560276985168}]
