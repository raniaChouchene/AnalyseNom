import spacy
from collections import defaultdict
import os
import networkx as nx
from itertools import combinations

# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")

def detect_character_interactions_in_folder(folder_path, output_file="character_interactions.txt"):
    """
    Détecte les interactions entre personnages (co-occurrences) dans tous les fichiers texte d'un dossier et construit un graphe des interactions.
    :param folder_path: Chemin vers le dossier contenant les fichiers texte d'entrée.
    :param output_file: Chemin vers le fichier texte de sortie.
    """
    # Initialiser un dictionnaire pour stocker les co-occurrences des personnages
    interactions = defaultdict(int)

    # Créer un graphe pour les interactions entre personnages
    G = nx.Graph()

    # Fonction pour vérifier si une entité est un personnage valide
    def is_valid_character(entity, context_sentence):
        if " " not in entity:  # Doit contenir un prénom et un nom
            return False
        if len(entity) < 3:  # Eviter les entités courtes
            return False
        if any(token.pos_ == "VERB" for token in context_sentence):  # Eviter les verbes
            return False
        return True

    # Parcourir tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Vérifier si c'est un fichier texte
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            print(f"Traitement du fichier : {filename}")
            
            # Lire le contenu du fichier d'entrée
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Analyse du texte avec spaCy
            doc = nlp(text)

            # Stockage des personnages présents dans ce fichier
            characters_in_file = []

            # Extraction des entités nommées de type "PER" (Personnes)
            for ent in doc.ents:
                if ent.label_ == "PER" and is_valid_character(ent.text, ent.sent):
                    characters_in_file.append(ent.text)

            # Ajouter les interactions entre les personnages présents dans ce fichier
            for char1, char2 in combinations(set(characters_in_file), 2):
                interactions[frozenset([char1, char2])] += 1
                # Ajouter les personnages dans le graphe s'ils n'y sont pas déjà
                if char1 not in G:
                    G.add_node(char1)
                if char2 not in G:
                    G.add_node(char2)
                # Ajouter une arête entre les personnages avec un poids correspondant au nombre de co-occurrences
                G.add_edge(char1, char2, weight=interactions[frozenset([char1, char2])])

    # Sauvegarder les résultats dans le fichier de sortie
    with open(output_file, 'a', encoding='utf-8') as file:  # Mode append pour ajouter sans écraser
        file.write("\n--- Interactions entre personnages ---\n")
        for interaction, count in sorted(interactions.items(), key=lambda x: x[1], reverse=True):
            # Récupérer les personnages des co-occurrences sous forme de tuple
            chars = list(interaction)
            file.write(f"{chars[0]} et {chars[1]}: {count} interactions\n")

    print(f"Détection des interactions terminée. Résultats ajoutés dans {output_file}")
    
    return G


# Exemple d'utilisation
folder_path = r"C:\Users\user\Desktop\M1ILSEN\AmsProjet3\DataFile"
G = detect_character_interactions_in_folder(folder_path)

# Visualiser le graphe des interactions
import matplotlib.pyplot as plt

# Dessiner le graphe
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=5000, node_color="skyblue", font_size=10, font_weight="bold", width=1, alpha=0.7)

# Ajouter les poids des arêtes (co-occurrences) sur le graphe
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Graphe des interactions entre personnages")
plt.show()
