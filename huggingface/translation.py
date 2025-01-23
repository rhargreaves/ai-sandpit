from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
while True:
    text = input("Enter English text: ")
    translation = translator(text)
    print(translation[0].get("translation_text"))
