import spacy
import os
from itertools import combinations
from collections import defaultdict
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


nlp = spacy.load("fr_core_news_sm")


def detect_and_draw_interactions(
    folder_path,
    graph_output="interactions_graph.png",
    csv_output="my_submission.csv",
    interaction_threshold=1
):
    """
    Détecte les interactions entre personnages dans des fichiers texte, génère des graphes GraphML et les exporte en CSV.
    
    :param folder_path: Chemin du dossier contenant les fichiers texte.
    :param graph_output: Chemin pour enregistrer une image de démonstration du graphe des interactions.
    :param csv_output: Chemin pour enregistrer le CSV avec les graphes au format GraphML.
    :param interaction_threshold: Nombre minimum d'interactions pour inclure dans le graphe.
    """
    interactions = defaultdict(lambda: defaultdict(int))

    def is_valid_character(entity):
        """
        Vérifie si une entité est un nom propre valide.
        """
        if len(entity.strip()) < 3: 
            return False
        doc = nlp(entity.strip())
        return all(token.pos_ == "PROPN" and token.is_alpha for token in doc)

    # Stockage des graphes pour l'exportation
    graph_data = {"ID": [], "graphml": []}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                if not text:
                    print(f"Fichier vide ignoré : {filename}")
                    continue

                # Analyse du texte avec spaCy
                doc = nlp(text)

                characters_in_text = [
                    ent.text.strip() for ent in doc.ents
                    if ent.label_ == "PER" and is_valid_character(ent.text)
                ]
                characters_in_text = list(set(characters_in_text))  # Éliminer les doublons

              
                for char1, char2 in combinations(characters_in_text, 2):
                    interactions[char1][char2] += 1
                    interactions[char2][char1] += 1

              
                G = nx.Graph()
                for char1, related_chars in interactions.items():
                    for char2, count in related_chars.items():
                        if count >= interaction_threshold:
                            G.add_edge(char1, char2, weight=count)

              
                for node in G.nodes():
                    G.nodes[node]["names"] = f"{node};{node.lower()}"

           
                graphml = "".join(nx.generate_graphml(G))
                graph_id = filename.split('.')[0]
                graph_data["ID"].append(graph_id)
                graph_data["graphml"].append(graphml)

            except Exception as e:
                print(f"Erreur lors du traitement du fichier {filename} : {e}")

  
    if G:
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, seed=42)
        edges = G.edges(data=True)

        nx.draw_networkx_nodes(G, pos, node_size=600, node_color="skyblue", alpha=0.9)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v, d in edges if d["weight"] >= interaction_threshold],
            width=[d["weight"] * 0.5 for u, v, d in edges if d["weight"] >= interaction_threshold],
            alpha=0.6,
            edge_color="gray"
        )
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        plt.title("Exemple de graphe d'interactions", fontsize=14)
        plt.savefig(graph_output, dpi=300)
        plt.show()

    if graph_data["ID"]:
        df = pd.DataFrame(graph_data)
        df.set_index("ID", inplace=True)
        df.to_csv(csv_output, encoding='utf-8')
        print(f"Exportation des graphes terminée. Fichier CSV enregistré sous {csv_output}")
    else:
        print("Aucun graphe valide à exporter.")


folder_path = r"C:\Users\user\Desktop\M1ILSEN\AmsProjet3\DataFile"

detect_and_draw_interactions(folder_path)
