import spacy
from collections import defaultdict
from itertools import combinations

# Load the French model
nlp = spacy.load("fr_core_news_sm")

# Function to dynamically detect characters and their possible aliases
def extract_character_mentions(input_file, output_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Process the text with Spacy
    doc = nlp(text)

    # Dictionary to keep track of character mentions
    character_mentions = defaultdict(int)

    # Open the output file for storing POS-tagged text if needed
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in doc.sents:
            # Filter and write POS tags if needed
            words_pos = [f"{token.text}/{token.pos_}" for token in sent if not token.is_punct]
            f.write(' '.join(words_pos) + '\n')

            # Named Entity Recognition (NER) for character identification
            for ent in sent.ents:
                if ent.label_ == "PER":  # 'PER' is the label for person entities
                    character_name = ent.text
                    character_mentions[character_name] += 1

    # Dynamic alias grouping by identifying similar names
    character_aliases = defaultdict(list)
    names_list = list(character_mentions.keys())
    
    for name1, name2 in combinations(names_list, 2):
        if name1.split()[-1] == name2.split()[-1]:  # Simple heuristic: same last name
            character_aliases[name1].append(name2)
            character_aliases[name2].append(name1)
    
    # Consolidate counts based on detected aliases
    consolidated_mentions = defaultdict(int)
    processed = set()
    for character, aliases in character_aliases.items():
        if character not in processed:
            total_count = character_mentions[character] + sum(character_mentions[alias] for alias in aliases)
            consolidated_mentions[character] = total_count
            processed.update([character] + aliases)

    # Write character mentions summary to a separate file
    with open("character_mentions_summary.txt", 'w', encoding='utf-8') as summary_file:
        for character, count in consolidated_mentions.items():
            summary_file.write(f"{character}: {count} occurrences\n")

# Paths to your input and output files
input_file = r'C:\Users\user\Desktop\AmsProjet3\clearTextFonadionSample.txt'
output_file = r'C:\Users\user\Desktop\AmsProjet3\output.txt'

# Call the function
extract_character_mentions(input_file, output_file)

print(f"Character extraction complete. Results saved in {output_file} and character summary saved in 'character_mentions_summary.txt'")
