import os
import re
from collections import defaultdict, Counter
import networkx as nx
import pandas as pd
from unidecode import unidecode
import spacy
from fuzzywuzzy import fuzz, process

# Load French spaCy model with larger vocabulary
nlp = spacy.load("fr_core_news_lg")


def sanitize_node_id(node_id):
    """Normalize node identifiers"""
    return unidecode(node_id.strip().replace(" ", "_").lower())


def advanced_name_matching(names, threshold=85):
    """Improve character name grouping"""
    grouped_characters = defaultdict(list)
    for name in names:
        if not grouped_characters:
            grouped_characters[name].append(name)
        else:
            # More sophisticated matching
            best_match, score = process.extractOne(
                name,
                list(grouped_characters.keys()),
                scorer=fuzz.token_sort_ratio  # More robust matching
            )
            if best_match and score > threshold:
                grouped_characters[best_match].append(name)
            else:
                grouped_characters[name].append(name)
    return grouped_characters


def detect_character_interactions(text, characters):
    """
    Advanced interaction detection with more context

    Args:
    - text (str): Chapter text
    - characters (dict): Grouped characters

    Returns:
    - nx.Graph: Interaction graph
    """
    G = nx.Graph()

    # Preprocess text for faster searching
    text_lower = text.lower()

    # Sentence-level interaction detection
    sentences = re.split(r'[.!?]+', text)

    # Track interaction strengths
    interaction_weights = Counter()

    for src, src_variants in characters.items():
        src_clean = sanitize_node_id(src)
        G.add_node(src_clean, names=";".join(src_variants))

        for tgt, tgt_variants in characters.items():
            if src != tgt:
                # Check interactions in sentences
                sentence_interactions = sum(
                    1 for sentence in sentences
                    if any(var.lower() in sentence.lower() for var in src_variants) and
                    any(var.lower() in sentence.lower() for var in tgt_variants)
                )

                # Add weighted edge if interactions exist
                if sentence_interactions > 0:
                    G.add_edge(src_clean, sanitize_node_id(tgt), weight=sentence_interactions)
                    interaction_weights[(src_clean, sanitize_node_id(tgt))] = sentence_interactions

    return G


def detect_and_draw_interactions(folder_path, output_csv="submission.csv"):
    books = [
        (list(range(1, 20)), "paf"),
        (list(range(1, 19)), "lca"),
    ]

    df_dict = {"ID": [], "graphml": []}

    for chapters, book_code in books:
        for chapter in chapters:
            repertory = "prelude_a_fondation" if book_code == "paf" else "les_cavernes_d_acier"
            chapter_file = f"{folder_path}/{repertory}/chapter_{chapter}.txt.preprocessed"

            if not os.path.isfile(chapter_file):
                print(f"Fichier introuvable: {chapter_file}")
                continue

            with open(chapter_file, "r", encoding="utf-8") as file:
                text = file.read()

            # Enhanced named entity recognition
            doc = nlp(text)

            # Extract characters with more robust filtering
            characters = [
                unidecode(ent.text.strip())
                for ent in doc.ents
                if ent.label_ == "PER" and
                   len(ent.text) > 2 and
                   not any(char.isdigit() for char in ent.text)  # Exclude names with numbers
            ]

            # Advanced name grouping
            grouped_characters = advanced_name_matching(characters)

            # Create interaction graph
            G = detect_character_interactions(text, grouped_characters)

            # Prepare submission
            df_dict["ID"].append(f"{book_code}{chapter - 1}")
            graphml = "".join(nx.generate_graphml(G))
            df_dict["graphml"].append(graphml)

    # Export submission
    df = pd.DataFrame(df_dict)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Soumission créée : {output_csv}")


# Lancer le script
folder_path = r"C:\Users\ADMIN\Desktop\AMS_PROJET\kaggle"
detect_and_draw_interactions(folder_path)