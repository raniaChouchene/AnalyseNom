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

# Configure logging
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
            name_match_threshold: int = 85,
            interaction_logging: bool = False
    ):
        """
        Initialize the Character Interaction Analyzer.

        Args:
            nlp_model (str): Spacy language model to use
            name_match_threshold (int): Fuzzy matching threshold for character names
            interaction_logging (bool): Enable detailed interaction logging
        """
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            logger.error(f"Could not load spaCy model {nlp_model}. Ensure it's installed.")
            raise

        self.name_match_threshold = name_match_threshold
        self.interaction_logging = interaction_logging

    @staticmethod
    def sanitize_node_id(node_id: str) -> str:
        """
        Normalize node identifiers consistently.

        Args:
            node_id (str): Original node identifier

        Returns:
            str: Normalized node identifier
        """
        return unidecode(node_id.strip().replace(" ", "_").lower())

    def advanced_name_matching(self, names: List[str]) -> Dict[str, List[str]]:
        """
        Group similar character names using fuzzy matching.

        Args:
            names (List[str]): List of character names to group

        Returns:
            Dict[str, List[str]]: Grouped character names
        """
        grouped_characters = defaultdict(list)
        for name in names:
            if not grouped_characters:
                grouped_characters[name].append(name)
            else:
                best_match, score = process.extractOne(
                    name,
                    list(grouped_characters.keys()),
                    scorer=fuzz.token_sort_ratio
                )
                if best_match and score > self.name_match_threshold:
                    grouped_characters[best_match].append(name)
                else:
                    grouped_characters[name].append(name)

        return grouped_characters

    def detect_character_interactions(
            self,
            text: str,
            characters: Dict[str, List[str]]
    ) -> nx.Graph:
        """
        Detect interactions between characters in a text.

        Args:
            text (str): Text to analyze
            characters (Dict[str, List[str]]): Grouped character names

        Returns:
            nx.Graph: Graph representing character interactions
        """
        G = nx.Graph()
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        interaction_weights = Counter()

        for src, src_variants in characters.items():
            src_clean = self.sanitize_node_id(src)
            G.add_node(src_clean, names=";".join(src_variants))

            for tgt, tgt_variants in characters.items():
                if src != tgt:
                    # Count sentence-level interactions
                    sentence_interactions = sum(
                        1 for sentence in sentences
                        if any(var.lower() in sentence.lower() for var in src_variants) and
                        any(var.lower() in sentence.lower() for var in tgt_variants)
                    )

                    # Add weighted edge if interactions exist
                    if sentence_interactions > 0:
                        G.add_edge(src_clean, self.sanitize_node_id(tgt), weight=sentence_interactions)
                        interaction_weights[(src_clean, self.sanitize_node_id(tgt))] = sentence_interactions

                    # Optional logging for detailed interaction tracking
                    if self.interaction_logging:
                        logger.debug(f"Interaction between {src_clean} and {tgt}: {sentence_interactions}")

        return G

    def process_chapters(
            self,
            folder_path: str,
            books: List[Tuple[List[int], str]] = None,
            output_csv: str = "submission.csv"
    ) -> pd.DataFrame:
        """
        Process chapters and generate interaction graphs.

        Args:
            folder_path (str): Base folder containing chapter files
            books (List[Tuple[List[int], str]], optional): Books and their chapter ranges
            output_csv (str): Path to save output CSV

        Returns:
            pd.DataFrame: DataFrame with chapter IDs and interaction graphs
        """
        # Default book configuration if not provided
        if books is None:
            books = [
                (list(range(1, 20)), "paf"),
                (list(range(1, 19)), "lca"),
            ]

        df_dict = {"ID": [], "graphml": []}

        for chapters, book_code in books:
            repertory = "prelude_a_fondation" if book_code == "paf" else "les_cavernes_d_acier"

            # Use tqdm for progress tracking
            for chapter in tqdm(chapters, desc=f"Processing {book_code} chapters"):
                chapter_file = os.path.join(folder_path, repertory, f"chapter_{chapter}.txt.preprocessed")

                try:
                    with open(chapter_file, "r", encoding="utf-8") as file:
                        text = file.read()

                    # Enhanced named entity recognition
                    doc = self.nlp(text)

                    # Extract characters with robust filtering
                    characters = [
                        unidecode(ent.text.strip())
                        for ent in doc.ents
                        if (ent.label_ == "PER" and
                            len(ent.text) > 2 and
                            not any(char.isdigit() for char in ent.text))
                    ]

                    # Advanced name grouping
                    grouped_characters = self.advanced_name_matching(characters)

                    # Create interaction graph
                    G = self.detect_character_interactions(text, grouped_characters)

                    # Prepare submission
                    df_dict["ID"].append(f"{book_code}{chapter - 1}")
                    graphml = "".join(nx.generate_graphml(G))
                    df_dict["graphml"].append(graphml)

                except FileNotFoundError:
                    logger.warning(f"Chapter file not found: {chapter_file}")
                except Exception as e:
                    logger.error(f"Error processing {chapter_file}: {e}")

        # Export submission
        df = pd.DataFrame(df_dict)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        logger.info(f"Submission created: {output_csv}")

        return df


def main():
    """Main execution function"""
    folder_path = r"C:\Users\ADMIN\Desktop\AMS_PROJET\kaggle"

    try:
        # Initialize analyzer with custom settings
        analyzer = CharacterInteractionAnalyzer(
            nlp_model="fr_core_news_lg",
            name_match_threshold=85,
            interaction_logging=False
        )

        # Process chapters and generate interaction graphs
        analyzer.process_chapters(folder_path)

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise


if __name__ == "__main__":
    main()