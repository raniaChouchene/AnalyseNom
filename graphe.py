import os
from collections import defaultdict
import networkx as nx
import pandas as pd
from unidecode import unidecode
import spacy
from fuzzywuzzy import fuzz, process
from xml.sax.saxutils import escape

# Charger le modèle spaCy français
nlp = spacy.load("fr_core_news_md")

# Fonction pour normaliser les noms de noeuds
def sanitize_node_id(node_id):
    return unidecode(node_id.strip().replace(" ", "_"))

# Fonction de détection et extraction
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

            doc = nlp(text)

            # Extraction des entités
            characters = []
            for ent in doc.ents:
                if ent.label_ == "PER" and len(ent.text) > 2:
                    characters.append(unidecode(ent.text.strip()))

            # Regroupement des noms similaires
            grouped_characters = defaultdict(list)
            for name in characters:
                if grouped_characters:
                    best_match, score = process.extractOne(
                        name, list(grouped_characters.keys()), scorer=fuzz.partial_ratio
                    )
                    if best_match and score > 85:
                        grouped_characters[best_match].append(name)
                    else:
                        grouped_characters[name].append(name)
                else:
                    grouped_characters[name].append(name)

            # Création du graphe
            G = nx.Graph()

            # Ajout des relations
            for src, src_variants in grouped_characters.items():
                src_clean = sanitize_node_id(src)
                G.add_node(src_clean, names=";".join(src_variants))
                for tgt, tgt_variants in grouped_characters.items():
                    if src != tgt:
                        if any(f" {var} " in text for var in src_variants) and any(f" {var} " in text for var in tgt_variants):
                            G.add_edge(src_clean, sanitize_node_id(tgt))

            # Sauvegarde dans le DataFrame
            df_dict["ID"].append(f"{book_code}{chapter - 1}")
            graphml = "".join(nx.generate_graphml(G))
            df_dict["graphml"].append(graphml)

    # Export du CSV
    df = pd.DataFrame(df_dict)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Soumission créée : {output_csv}")


# Lancer le script
folder_path = r"C:\Users\ADMIN\Desktop\AMS_PROJET\kaggle"
detect_and_draw_interactions(folder_path)
