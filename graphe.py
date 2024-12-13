import os
import re
import logging
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import networkx as nx
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
            context_window: int = 3  # Réduction de la fenêtre de contexte
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

    def prettify_graphml(self, graphml: str) -> str:
        """
        Formate le GraphML pour assurer la compatibilité.
        """
        try:
            root = ET.fromstring(graphml)

            # Suppression des attributs potentiellement problématiques
            for elem in root.iter():
                if 'yfiles' in str(elem.tag):
                    root.remove(elem)

                # Nettoyage des attributs
                attrs_to_remove = [
                    attr for attr in elem.attrib
                    if attr.startswith('{') or
                       any(x in attr for x in ['xmlns:', 'xsi:', 'yfiles'])
                ]
                for attr in attrs_to_remove:
                    del elem.attrib[attr]

            # Conversion en chaîne formatée
            rough_string = ET.tostring(root, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")

        except Exception as e:
            logger.error(f"Erreur lors du formatage GraphML : {e}")
            return graphml

    def process_chapters(
            self,
            folder_path: str,
            books: List[Tuple[List[int], str]] = None,
            output_csv: str = "submission.csv"
    ) -> pd.DataFrame:
        """
        Traite les chapitres et génère des graphes d'interactions.
        """
        # Configuration par défaut des livres
        if books is None:
            books = [
                (list(range(1, 20)), "paf"),
                (list(range(1, 19)), "lca"),
            ]

        df_dict = {"ID": [], "graphml": []}

        for chapters, book_code in books:
            repertory = "prelude_a_fondation" if book_code == "paf" else "les_cavernes_d_acier"

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

                    # Création du graphe
                    G = self.detect_character_interactions(text, grouped_characters)

                    # Génération du GraphML
                    graphml_str = nx.generate_graphml(G)

                    # Formatage et nettoyage du GraphML
                    graphml_cleaned = "".join(graphml_str)
                    graphml_prettified = self.prettify_graphml(graphml_cleaned)

                    # Préparation de la soumission
                    df_dict["ID"].append(f"{book_code}{chapter - 1}")
                    df_dict["graphml"].append(graphml_prettified)

                except FileNotFoundError:
                    logger.warning(f"Fichier de chapitre non trouvé : {chapter_file}")
                    # Ajout d'une entrée vide pour maintenir la cohérence
                    df_dict["ID"].append(f"{book_code}{chapter - 1}")
                    df_dict["graphml"].append("")
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de {chapter_file}: {e}")
                    # Ajout d'une entrée vide pour maintenir la cohérence
                    df_dict["ID"].append(f"{book_code}{chapter - 1}")
                    df_dict["graphml"].append("")

        # Exportation de la soumission
        df = pd.DataFrame(df_dict)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        logger.info(f"Soumission créée : {output_csv}")

        return df


def main():
    """Fonction d'exécution principale"""
    folder_path = r"C:\Users\ADMIN\Desktop\AMS_PROJET\kaggle"

    try:
        # Initialisation de l'analyseur
        analyzer = CharacterInteractionAnalyzer(
            nlp_model="fr_core_news_lg",
            name_match_threshold=90,
            interaction_logging=False,
            context_window=3  # Fenêtre de contexte réduite
        )

        # Traitement des chapitres
        analyzer.process_chapters(folder_path)

    except Exception as e:
        logger.error(f"Une erreur s'est produite lors de l'exécution : {e}")
        raise


if __name__ == "__main__":
    main()