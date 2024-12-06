import spacy
import os
from itertools import combinations
from collections import defaultdict
import networkx as nx
import pandas as pd

# Charger le modèle de langue français de spaCy
nlp = spacy.load("fr_core_news_sm")

def detect_and_draw_interactions(
    folder_path,
    output_csv="submission.csv",
    interaction_threshold=1
):
    """
    Détecte les interactions entre les personnages dans des fichiers texte, génère un graphe pour chaque chapitre et exporte les résultats sous forme de CSV pour Kaggle.
    
    :param folder_path: Chemin du dossier contenant les fichiers texte.
    :param output_csv: Chemin pour enregistrer le fichier CSV des résultats.
    :param interaction_threshold: Nombre minimum d'interactions pour inclure dans le graphe et le CSV.
    """
    interactions = defaultdict(lambda: defaultdict(int))

    def is_valid_character(entity):
        """
        Vérifie si une entité est un nom propre valide.
        """
        if len(entity.strip()) < 3:  # Élimine les noms trop courts
            return False
        doc = nlp(entity.strip())
        return all(token.pos_ == "PROPN" and token.is_alpha for token in doc)

    # Définir les chapitres à analyser
    chapter_files = [
        (range(0, 19), "paf"), 
       # (range(0, 18), "lca")   # 18 chapitres pour "Les Cavernes d'Acier"
    ]
    
    df_dict = {"ID": [], "graphml": []}

    # Traiter chaque chapitre des livres
    for chapters, book_code in chapter_files:
        for chapter in chapters:
            file_path = os.path.join(folder_path, f"chapter_{chapter+1}.txt")  # Compte à partir de 1 pour les fichiers
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read().strip()
                    if not text:
                        print(f"Fichier vide ignoré : {file_path}")
                        continue

                    # Analyse du texte avec spaCy
                    doc = nlp(text)

                    # Extraire les noms de personnages valides
                    characters_in_text = [
                        ent.text.strip() for ent in doc.ents
                        if ent.label_ == "PER" and is_valid_character(ent.text)
                    ]
                    characters_in_text = list(set(characters_in_text))  # Éliminer les doublons

                    # Créer les interactions entre les personnages
                    for char1, char2 in combinations(characters_in_text, 2):
                        interactions[char1][char2] += 1
                        interactions[char2][char1] += 1

                    # Créer un graphe pour le chapitre
                    G = nx.Graph()
                    for char1, related_chars in interactions.items():
                        for char2, count in related_chars.items():
                            if count >= interaction_threshold:
                                G.add_edge(char1, char2, weight=count)

                    # Ajouter les noms des personnages au graphe sous l'attribut 'names'
                    for node in G.nodes():
                        G.nodes[node]["names"] = node  # Ajout du nom du personnage à l'attribut 'names'

                    # Enregistrer le graphe au format GraphML
                    graphml = "".join(nx.generate_graphml(G))

                    # Ajouter les résultats dans le DataFrame
                    df_dict["ID"].append(f"{book_code}{chapter}")
                    df_dict["graphml"].append(graphml)

                except Exception as e:
                    print(f"Erreur lors du traitement du fichier {file_path} : {e}")
            else:
                print(f"Fichier non trouvé : {file_path}")

  

    # Créer le DataFrame et l'exporter en CSV
    if df_dict["ID"]:
        df = pd.DataFrame(df_dict)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Soumission terminée. Fichier CSV enregistré sous {output_csv}")
    else:
        print("Aucune interaction détectée. Aucun fichier CSV n'a été créé.")

# Chemin du dossier contenant les fichiers texte
folder_path = r"C:\Users\user\Desktop\M1ILSEN\AmsProjet3\DataFile"

# Exécuter la fonction
detect_and_draw_interactions(folder_path)
