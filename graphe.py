import spacy
import os
from itertools import combinations
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")

def detect_and_draw_interactions(folder_path, output_file="interactions_summary.txt", graph_output="interactions_graph.png", csv_output="interactions.csv"):
    """
    Détecte les interactions entre personnages, génère un graphique de réseau et exporte les données au format CSV.
    :param folder_path: Chemin vers le dossier contenant les fichiers texte d'entrée.
    :param output_file: Chemin vers le fichier texte de sortie.
    :param graph_output: Chemin vers l'image de sortie du graphique.
    :param csv_output: Chemin vers le fichier CSV de sortie.
    """
    interactions = defaultdict(lambda: defaultdict(int))

    def is_valid_character(entity):
        """
        Vérifie si une entité est un personnage valide (nom propre uniquement).
        """
        if len(entity) < 3:
            return False
        doc = nlp(entity)
        if all(token.pos_ == "PROPN" and token.is_alpha for token in doc):
            return True
        return False

    # Parcourir les fichiers texte dans le dossier
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Analyse le texte
            doc = nlp(text)

            # Extraction des personnages
            characters_in_text = [
                ent.text for ent in doc.ents
                if ent.label_ == "PER" and is_valid_character(ent.text)
            ]

            characters_in_text = list(set(characters_in_text))

            # Création des interactions entre personnages
            for char1, char2 in combinations(characters_in_text, 2):
                interactions[char1][char2] += 1
                interactions[char2][char1] += 1

    # Vérifier si des interactions ont été trouvées
    if not interactions:
        print("Aucune interaction détectée. Aucun fichier ne sera créé.")
        return

    # Écrire les interactions dans un fichier texte
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("\n--- Interactions entre personnages ---\n")
        for char1, related_chars in interactions.items():
            for char2, count in related_chars.items():
                if char1 < char2:
                    file.write(f"{char1} <-> {char2}: {count} interactions\n")

    print(f"Extraction des interactions terminée. Résultats ajoutés dans {output_file}")

    # Création et visualisation du graphe
    G = nx.Graph()
    for char1, related_chars in interactions.items():
        for char2, count in related_chars.items():
            if count > 0:
                G.add_edge(char1, char2, weight=count)

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    plt.title("Interactions entre personnages")
    plt.savefig(graph_output)
    plt.show()

    # Exporter les interactions au format CSV
    data = []
    for char1, related_chars in interactions.items():
        for char2, count in related_chars.items():
            if char1 < char2:
                data.append({"Personnage 1": char1, "Personnage 2": char2, "Interactions": count})

    if data:
        df = pd.DataFrame(data)
        df.to_csv(csv_output, index=False, encoding='utf-8')
        print(f"Export des interactions terminé. Fichier CSV sauvegardé sous {csv_output}")
    else:
        print("Aucune donnée valide pour le fichier CSV.")

# Appel de la fonction
folder_path = r"C:\Users\user\Desktop\M1ILSEN\AmsProjet3\DataFile"
detect_and_draw_interactions(folder_path)
