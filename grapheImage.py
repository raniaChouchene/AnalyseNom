import os
import re
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from unidecode import unidecode
from fuzzywuzzy import fuzz, process
from tqdm import tqdm
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

# Configuration de la journalisation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CharacterInteractionAnalyzer:
    def __init__(
            self,
            nlp_model: str = "fr_core_news_lg",
            name_match_threshold: int = 90,
            interaction_logging: bool = False,
            context_window: int = 3
    ):
        """
        Initialise l'Analyseur d'Interactions de Personnages.
        """
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            logger.error(f"Impossible de charger le modèle spaCy {nlp_model}.")
            raise

        self.name_match_threshold = name_match_threshold
        self.interaction_logging = interaction_logging
        self.context_window = context_window

    @staticmethod
    def sanitize_node_id(node_id: str) -> str:
        """Normalise les identifiants de nœuds."""
        return unidecode(node_id.strip().replace(" ", "_").lower())

    def advanced_name_matching(self, names: List[str]) -> Dict[str, List[str]]:
        """
        Groupe les noms de personnages similaires.
        """
        grouped_characters = defaultdict(list)
        unique_names = list(set(names))  # Éliminer les doublons

        for name in unique_names:
            cleaned_name = name.strip().lower()
            if not grouped_characters:
                grouped_characters[cleaned_name].append(name)
            else:
                best_match, score = process.extractOne(
                    cleaned_name,
                    list(grouped_characters.keys()),
                    scorer=fuzz.token_set_ratio
                )
                if best_match and score > self.name_match_threshold:
                    grouped_characters[best_match].append(name)
                else:
                    grouped_characters[cleaned_name].append(name)

        return grouped_characters

    def detect_character_interactions(
            self,
            text: str,
            characters: Dict[str, List[str]]
    ) -> nx.Graph:
        """
        Détecte les interactions entre personnages.
        """
        G = nx.Graph()
        sentences = re.split(r'[.!?]+', text)

        # Conversion des noms en identifiants normalisés
        character_nodes = {self.sanitize_node_id(src): src_variants
                           for src, src_variants in characters.items()}

        for src_clean, src_variants in character_nodes.items():
            # Ajout des nœuds
            G.add_node(src_clean, names=";".join(src_variants))

            for tgt_clean, tgt_variants in character_nodes.items():
                if src_clean != tgt_clean:
                    # Calcul des interactions
                    sentence_interactions = sum(
                        1 for sentence in sentences
                        if any(var.lower() in sentence.lower() for var in src_variants) and
                        any(var.lower() in sentence.lower() for var in tgt_variants)
                    )

                    # Ajout des arêtes avec un poids
                    if sentence_interactions > 0:
                        G.add_edge(src_clean, tgt_clean, weight=sentence_interactions)

        return G

    def visualize_character_interactions(
            self,
            folder_path: str,
            book_code: str,
            chapters: List[int],
            output_path: str = "character_interactions.png"
    ):
        """
        Visualise les interactions entre personnages à partir des chapitres spécifiés.
        """
        # Répertoire du livre
        repertory = "prelude_a_fondation" if book_code == "paf" else "les_cavernes_d_acier"

        # Graphe combiné pour tous les chapitres
        combined_graph = nx.Graph()

        for chapter in tqdm(chapters, desc=f"Traitement des chapitres de {book_code}"):
            chapter_file = os.path.join(folder_path, repertory, f"chapter_{chapter}.txt.preprocessed")

            try:
                with open(chapter_file, "r", encoding="utf-8") as file:
                    text = file.read()

                # Reconnaissance des entités nommées
                doc = self.nlp(text)

                # Extraction des personnages
                characters = [
                    unidecode(ent.text.strip())
                    for ent in doc.ents
                    if (ent.label_ == "PER" and
                        len(ent.text) > 2 and
                        not any(char.isdigit() for char in ent.text))
                ]

                # Regroupement des noms
                grouped_characters = self.advanced_name_matching(characters)

                # Création du graphe pour ce chapitre
                chapter_graph = self.detect_character_interactions(text, grouped_characters)

                # Fusion des graphes
                combined_graph = nx.compose(combined_graph, chapter_graph)

            except FileNotFoundError:
                logger.warning(f"Fichier de chapitre non trouvé : {chapter_file}")
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {chapter_file}: {e}")

        # Visualisation
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(combined_graph, k=0.5, iterations=50)

        # Dessin des nœuds
        nx.draw_networkx_nodes(
            combined_graph,
            pos,
            node_color='lightblue',
            node_size=[combined_graph.degree(node) * 300 for node in combined_graph.nodes()],
            alpha=0.8
        )

        # Dessin des arêtes
        nx.draw_networkx_edges(
            combined_graph,
            pos,
            width=[combined_graph[u][v]['weight'] * 0.5 for (u, v) in combined_graph.edges()],
            alpha=0.5
        )

        # Étiquettes des nœuds
        labels = {node: node for node in combined_graph.nodes()}
        nx.draw_networkx_labels(combined_graph, pos, labels, font_size=10)

        plt.title(f"Interactions des personnages - {book_code.upper()}")
        plt.axis('off')

        # Sauvegarde du graphique
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Graphique d'interactions sauvegardé : {output_path}")

        return combined_graph


def main():
    """Fonction d'exécution principale"""
    folder_path = r"C:\Users\ADMIN\Desktop\AMS_PROJET\kaggle"

    try:
        # Initialisation de l'analyseur
        analyzer = CharacterInteractionAnalyzer(
            nlp_model="fr_core_news_lg",
            name_match_threshold=90,
            interaction_logging=False,
            context_window=3
        )

        # Visualisation des interactions pour différents livres et chapitres
        # Prélude à Fondation (PAF)
        analyzer.visualize_character_interactions(
            folder_path,
            "paf",
            list(range(1, 20)),
            output_path="paf_character_interactions.png"
        )

        # Les Cavernes d'Acier (LCA)
        analyzer.visualize_character_interactions(
            folder_path,
            "lca",
            list(range(1, 19)),
            output_path="lca_character_interactions.png"
        )

    except Exception as e:
        logger.error(f"Une erreur s'est produite lors de l'exécution : {e}")
        raise


if __name__ == "__main__":
    main()