from collections import Counter
import matplotlib.pyplot as plt

with open("dataset.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokens = text.split()
total_tokens = len(tokens)

punctuation_marks = list(".,?")
punct_counter = Counter()

for char in text:
    if char in punctuation_marks:
        punct_counter[char] += 1

total_punct = sum(punct_counter.values())

print(f"Total Tokens: {total_tokens}")
print(f"Total Punctuations: {total_punct}\n")

print("Punctuation Frequency:")
for punct, count in punct_counter.most_common():
    print(f"'{punct}': {count} ({(count/total_punct)*100:.2f}%)")

print(f"\nPunctuation-to-token ratio: {total_punct / total_tokens:.2f}")

labels = list(punct_counter.keys())
counts = [punct_counter[p] for p in labels]
percentages = [(c / total_punct) * 100 for c in counts]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, counts, color='skyblue', edgecolor='black')
plt.title("Punctuation Frequency Distribution")
plt.xlabel("Punctuation Mark")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.5)

for bar, pct in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{pct:.1f}%", 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
