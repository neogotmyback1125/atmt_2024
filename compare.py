import pandas as pd
import sacrebleu
import matplotlib.pyplot as plt
import os
from sacremoses import MosesDetokenizer

# Function to load translations with error handling for encoding issues
def load_translations(file_path):
    with open(file_path, "r", encoding="ISO-8859-1", errors="replace") as f:
        return [line.strip() for line in f]

# Function to detokenize translations
def detokenize_translations(input_file, output_file):
    detokenizer = MosesDetokenizer(lang="en")
    with open(input_file, "r", encoding="ISO-8859-1", errors="replace") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            detokenized_line = detokenizer.detokenize(line.strip().split())
            outfile.write(detokenized_line + "\n")

# Paths to translation files and their detokenized versions
beam_sizes = [1, 3, 5, 10, 15, 20, 25]
translations = {}

for beam in beam_sizes:
    original_file = f"model_translations_beam_{beam}.txt"
    detokenized_file = f"postprocessed_model_translations_beam_{beam}.txt"
    if os.path.exists(original_file):
        # Detokenize the file
        detokenize_translations(original_file, detokenized_file)
        translations[beam] = load_translations(detokenized_file)
    else:
        print(f"File {original_file} does not exist.")
        translations[beam] = []

# Convert to DataFrame for easier comparison
df_translations = pd.DataFrame(translations)
df_translations.columns = [f"Beam {beam}" for beam in beam_sizes]

# Display the first few rows of translations
print(df_translations.head())

# Compare first 5 sentences across beam sizes
for i in range(5):
    print(f"Sentence {i+1}:")
    for beam in beam_sizes:
        print(f"Beam {beam}: {df_translations[f'Beam {beam}'][i]}")
    print("-" * 50)

# Load reference translations with error handling
reference_file = "data/en-fr/prepared/test.fr"
if os.path.exists(reference_file):
    with open(reference_file, "r", encoding="ISO-8859-1", errors="replace") as ref:
        references = [line.strip() for line in ref]
else:
    raise FileNotFoundError(f"Reference file {reference_file} does not exist.")

# Compute BLEU scores for each beam size
bleu_scores = {}
for beam in beam_sizes:
    hypothesis = translations[beam]
    bleu = sacrebleu.corpus_bleu(hypothesis, [references])
    bleu_scores[beam] = bleu.score

# Convert BLEU scores to a DataFrame
df_bleu = pd.DataFrame(list(bleu_scores.items()), columns=["Beam Size", "BLEU Score"])
print(df_bleu)

# Add decoding times (replace with actual values)
decoding_times = [5, 214, 15, 50, 90, 160, 230]  # Updated decoding times for beam size 3
df_bleu["Decoding Time (s)"] = decoding_times

# Display BLEU scores and decoding times
print(df_bleu)

# Visualize BLEU scores and decoding times
# BLEU Score Plot
plt.figure()
plt.plot(df_bleu["Beam Size"], df_bleu["BLEU Score"], marker="o", label="BLEU Score")
plt.xlabel("Beam Size")
plt.ylabel("BLEU Score")
plt.title("BLEU Score vs Beam Size")
plt.grid()
plt.legend()
plt.show()

# Decoding Time Plot
beam_sizes_origin = [1, 5, 10, 15, 20, 25]
decoding_times_origin = [5, 15, 50, 90, 160, 230]

beam_sizes_new = [3]
decoding_times_new = [214]

plt.plot(beam_sizes_origin, decoding_times_origin, marker="o", label="Original Beam Sizes", color="blue")
plt.scatter(beam_sizes_new, decoding_times_new, color="red", label="Beam Size 3", s=100, zorder=5)

plt.xlabel("Beam Size")
plt.ylabel("Decoding Time (seconds)")
plt.title("Decoding Time vs Beam Size (Updated)")
plt.legend()
plt.grid(True)
plt.show()
