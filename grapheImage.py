import os
import re
import logging
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import spacy
from unidecode import unidecode
from fuzzywuzzy import fuzz, process
from tqdm import tqdm
from PyPDF2 import PdfReader

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
            context_window: int = 3,
            anti_dic_path: str = r"C:\Users\user\Desktop\M1ILSEN\AmsProjet3\antiDic.txt"
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

        # Chargement du dictionnaire des mots à éviter
        self.anti_dic = self.load_anti_dic(anti_dic_path)

        # Dictionnaires pour les polarités
        self.friendship_verbs = {"aimer", "adorer", "soutenir", "aider", "protéger", "défendre"}
        self.enmity_verbs = {"détester", "haïr", "combattre", "trahir", "attaquer", "blesser"}
        self.neutral_verbs = {"parler", "voir", "rencontrer", "observer", "écouter"}

        self.friendship_adjectives = {"gentil", "sympathique", "bienveillant", "loyal"}
        self.enmity_adjectives = {"méchant", "hostile", "dangereux", "traître"}

    def load_anti_dic(self, anti_dic_path: str) -> Set[str]:
        """Charge les mots à éviter depuis un fichier."""
        with open(anti_dic_path, 'r', encoding='utf-8') as file:
            return set(line.strip().lower() for line in file)

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
            if cleaned_name in self.anti_dic:
                continue  # Ignorer les mots à éviter
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

    def detect_polarity(self, sentence: str) -> int:
        """
        Détermine la polarité d'une phrase en fonction des verbes et adjectifs.
        """
        doc = self.nlp(sentence)
        polarity = 0

        for token in doc:
            if token.lemma_ in self.friendship_verbs:
                polarity += 1
            elif token.lemma_ in self.enmity_verbs:
                polarity -= 1
            elif token.lemma_ in self.neutral_verbs:
                polarity += 0

            if token.pos_ == "ADJ":
                if token.lemma_ in self.friendship_adjectives:
                    polarity += 1
                elif token.lemma_ in self.enmity_adjectives:
                    polarity -= 1

        # Limite la polarité à l'intervalle [-3, 3]
        return max(-3, min(3, polarity))

    def detect_character_interactions(
            self,
            text: str,
            characters: Dict[str, List[str]]
    ) -> nx.Graph:
        """
        Détecte les interactions entre personnages avec polarités.
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
                    # Calcul des interactions et polarités
                    total_polarity = 0
                    interaction_count = 0

                    for sentence in sentences:
                        if (any(var.lower() in sentence.lower() for var in src_variants) and
                            any(var.lower() in sentence.lower() for var in tgt_variants)):
                            polarity = self.detect_polarity(sentence)
                            total_polarity += polarity
                            interaction_count += 1

                    # Ajout des arêtes avec un poids et une polarité
                    if interaction_count > 0:
                        average_polarity = total_polarity / interaction_count
                        G.add_edge(src_clean, tgt_clean, weight=interaction_count, polarity=average_polarity)

        return G

    def find_friendship_cliques(self, graph: nx.Graph) -> List[List[str]]:
        """
        Trouve des sous-réseaux d'amitié (cliques) dans le graphe.
        """
        cliques = list(nx.find_cliques(graph))
        return cliques

    def rank_characters_by_popularity(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Classe les personnages par popularité (centralité de degré).
        """
        return nx.degree_centrality(graph)

    def visualize_character_interactions(
            self,
            folder_path: str,
            output_path: str = "character_interactions.png"
    ):
        """
        Visualise les interactions entre personnages à partir des fichiers PDF.
        """
        # Graphe combiné pour tous les fichiers
        combined_graph = nx.Graph()

        # Lister tous les fichiers PDF dans le répertoire
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

        for pdf_file in tqdm(pdf_files, desc="Traitement des fichiers PDF"):
            file_path = os.path.join(folder_path, pdf_file)

            try:
                # Lire le contenu du fichier PDF
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

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

                # Création du graphe pour ce fichier
                file_graph = self.detect_character_interactions(text, grouped_characters)

                # Fusion des graphes
                combined_graph = nx.compose(combined_graph, file_graph)

            except Exception as e:
                logger.error(f"Erreur lors du traitement de {pdf_file}: {e}")

        # Trouver les cliques d'amitié
        cliques = self.find_friendship_cliques(combined_graph)
        logger.info(f"Sous-réseaux d'amitié (cliques) : {cliques}")

        # Classer les personnages par popularité
        popularity_ranking = self.rank_characters_by_popularity(combined_graph)
        logger.info(f"Classement des personnages par popularité : {popularity_ranking}")

        # Visualisation avec matplotlib
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

        # Dessin des arêtes avec des couleurs et largeurs basées sur la polarité
        edge_colors = []
        edge_widths = []
        for u, v, data in combined_graph.edges(data=True):
            polarity = data.get("polarity", 0)
            if polarity > 0:
                edge_colors.append("green")  # Amitié
            elif polarity < 0:
                edge_colors.append("red")  # Inimitié
            else:
                edge_colors.append("gray")  # Neutre
            edge_widths.append(abs(polarity) * 2)  # Largeur proportionnelle à la polarité

        nx.draw_networkx_edges(
            combined_graph,
            pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6
        )

        # Étiquettes des nœuds
        labels = {node: node for node in combined_graph.nodes()}
        nx.draw_networkx_labels(combined_graph, pos, labels, font_size=10)

        plt.title("Interactions des personnages - Corpus ASIMOV")
        plt.axis('off')

        # Sauvegarde du graphique
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Graphique d'interactions sauvegardé : {output_path}")

        return combined_graph


def main():
    """Fonction d'exécution principale"""
    folder_path = r"C:\Users\user\Desktop\M1ILSEN\AmsProjet3\Corpus_ASIMOV"

    try:
        # Initialisation de l'analyseur
        analyzer = CharacterInteractionAnalyzer(
            nlp_model="fr_core_news_lg",
            name_match_threshold=90,
            interaction_logging=False,
            context_window=3
        )

        # Visualisation des interactions pour tous les fichiers PDF
        analyzer.visualize_character_interactions(
            folder_path,
            output_path="ASIMOV_character_interactions3.png"
        )

    except Exception as e:
        logger.error(f"Une erreur s'est produite lors de l'exécution : {e}")
        raise


if __name__ == "__main__":
    main()