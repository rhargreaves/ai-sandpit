from transformers import pipeline

with open("summarization-input.txt", "r") as f:
    text = f.read()
    summarizer = pipeline("summarization")
    summary = summarizer(text, min_length=200, max_length=300)
    print(summary[0].get("summary_text"))
