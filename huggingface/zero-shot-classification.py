from transformers import pipeline

classifier = pipeline("zero-shot-classification")
while True:
    text = input("Enter text: ")
    candidate_labels = ["education", "politics", "gaming", "sports", "nature"]
    result = classifier(text, candidate_labels=candidate_labels)
    print()

    for label, score in zip(result['labels'], result['scores']):
        print(f"{label}: {score:.2f}")

# Some interesting results:

# guns
# ----
# gaming: 0.31
# sports: 0.26
# politics: 0.18
# education: 0.14
# nature: 0.12

# clay pigeon shooting
# --------------------
# sports: 0.75
# nature: 0.19
# gaming: 0.04
# education: 0.01
# politics: 0.01
