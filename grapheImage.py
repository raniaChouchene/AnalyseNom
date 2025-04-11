import os
import re
import logging
import sys
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from unidecode import unidecode
from fuzzywuzzy import fuzz, process
from tqdm import tqdm
from PyPDF2 import PdfReader
from pyvis.network import Network

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
            interaction_logging: bool = False,
            context_window: int = 3,
            anti_dic_path: str = r"C:\Program Files\Python 3.12\antiDic.txt",
            min_name_length: int = 3
    ):
        """
        Initialize the Character Interaction Analyzer with robust spaCy handling.
        """
        # Initialize NLP model with proper error handling
        self.nlp = self._load_spacy_model(nlp_model)
        if self.nlp is None:
            raise RuntimeError("Failed to initialize spaCy model")

        self.name_match_threshold = name_match_threshold
        self.interaction_logging = interaction_logging
        self.context_window = context_window
        self.min_name_length = min_name_length

        # Load anti-dictionary
        self.anti_dic = self._load_anti_dic(anti_dic_path)

        # Initialize polarity dictionaries 
        self._init_polarity_dictionaries()

    def _load_spacy_model(self, model_name: str):
        """Load spaCy model with installation fallback."""
        try:
            import spacy
            try:
                return spacy.load(model_name)
            except OSError:
                logger.warning(f"spaCy model '{model_name}' not found. Attempting to download...")
                from spacy.cli import download
                download(model_name)
                return spacy.load(model_name)
        except ImportError:
            logger.error("spaCy not installed. Please install with: pip install spacy")
            return None
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            return None

    def _load_anti_dic(self, path: str) -> Set[str]:
        """Load anti-dictionary with error handling."""
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return {unidecode(line.strip().lower()) for line in f}
            return set()
        except Exception as e:
            logger.warning(f"Could not load anti-dictionary: {e}")
            return set()

    def _init_polarity_dictionaries(self):
        """Initialize polarity dictionaries."""
        self.friendship_verbs = {
            "aimer", "adorer", "soutenir", "aider", "protéger", "défendre",
            "sauver", "réconforter", "féliciter", "encourager", "apprécier",
            "embrasser", "câliner", "sourire", "complimenter", "secourir"
        }
        self.enmity_verbs = {
            "détester", "haïr", "combattre", "trahir", "attaquer", "blesser",
            "tuer", "frapper", "insulter", "menacer", "voler", "tromper",
            "humilier", "piéger", "jalouser", "poignarder", "dénoncer"
        }
        self.neutral_verbs = {
            "parler", "voir", "rencontrer", "observer", "écouter", "dire",
            "demander", "répondre", "regarder", "trouver", "penser", "croire"
        }
        self.friendship_adjectives = {
            "gentil", "sympathique", "bienveillant", "loyal", "aimable",
            "généreux", "honnête", "fidèle", "chaleureux", "dévoué", "courageux"
        }
        self.enmity_adjectives = {
            "méchant", "hostile", "dangereux", "traître", "cruel",
            "égoïste", "fourbe", "malveillant", "violent", "haineux", "sadique"
        }

    def is_valid_character_name(self, name: str) -> bool:
        """Check if a name is valid for a character."""
        if not name or len(name) < self.min_name_length:
            return False
        
        name_lower = unidecode(name.lower())
        if any(char.isdigit() for char in name):
            return False
            
        # Exclude generic terms
        generic_terms = {"monsieur", "madame", "docteur", "professeur", "capitaine", "s absolument"}
        first_word = name_lower.split()[0] if name_lower.split() else ""
        if first_word in generic_terms:
            return False
            
        return name_lower not in self.anti_dic

    def normalize_name(self, name: str) -> str:
        """Normalize character names."""
        # Remove titles and punctuation
        name = re.sub(r"(^|\s)(M|Mme|Dr|Pr)\.?\s+", " ", name, flags=re.IGNORECASE)
        name = re.sub(r"[^\w\s]", "", name)
        return unidecode(name.strip().replace("_", " ").lower())

    def advanced_name_matching(self, names: List[str]) -> Dict[str, List[str]]:
        """Group similar character names."""
        cleaned_names = [self.normalize_name(name) for name in names if self.is_valid_character_name(name)]
        unique_names = list(set(cleaned_names))
        
        grouped = defaultdict(list)
        processed = set()

        for name in unique_names:
            if name in processed:
                continue
                
            # Find similar names not yet processed
            similar = [n for n in unique_names 
                      if n not in processed 
                      and fuzz.token_set_ratio(name, n) > self.name_match_threshold]
            
            if similar:
                canonical = max(similar, key=len)
                grouped[canonical] = similar
                processed.update(similar)

        return grouped

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with error handling."""
        try:
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""

    def extract_characters(self, text: str) -> Dict[str, List[str]]:
        """Extract characters from text using NER."""
        doc = self.nlp(text)
        characters = [
            ent.text.strip() 
            for ent in doc.ents 
            if ent.label_ == "PER" and self.is_valid_character_name(ent.text)
        ]
        return self.advanced_name_matching(characters)

    def detect_polarity(self, sentence: str) -> int:
        """Detect relationship polarity in a sentence."""
        doc = self.nlp(sentence)
        polarity = 0

        for token in doc:
            lemma = token.lemma_.lower()
            if token.pos_ == "VERB":
                if lemma in self.friendship_verbs:
                    polarity += 1
                elif lemma in self.enmity_verbs:
                    polarity -= 1
            elif token.pos_ == "ADJ":
                if lemma in self.friendship_adjectives:
                    polarity += 1
                elif lemma in self.enmity_adjectives:
                    polarity -= 1

        return max(-3, min(3, polarity))

    def detect_interactions(self, text: str, characters: Dict[str, List[str]]) -> Tuple[nx.Graph, List[tuple]]:
        """Detect character interactions."""
        G = nx.Graph()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        relations = []

        # Add nodes
        for canon, aliases in characters.items():
            G.add_node(canon, aliases=", ".join(aliases), size=10)

        # Analyze interactions
        for src, src_aliases in characters.items():
            for tgt, tgt_aliases in characters.items():
                if src != tgt:
                    interactions = []
                    
                    for sent in sentences:
                        src_in = any(alias.lower() in sent.lower() for alias in src_aliases)
                        tgt_in = any(alias.lower() in sent.lower() for alias in tgt_aliases)
                        
                        if src_in and tgt_in:
                            polarity = self.detect_polarity(sent)
                            interactions.append((sent, polarity))
                            
                    if interactions:
                        total_weight = len(interactions)
                        avg_polarity = sum(p for _, p in interactions) / total_weight
                        rel_type = self._get_relation_type(avg_polarity)
                        
                        G.add_edge(src, tgt, 
                                 weight=total_weight,
                                 polarity=avg_polarity,
                                 type=rel_type)
                        
                        relations.append((
                            src, tgt, 
                            total_weight, 
                            rel_type, 
                            "; ".join(s[:50]+"..." for s, _ in interactions[:3])
                        ))

        return G, relations

    def _get_relation_type(self, polarity: float) -> str:
        """Classify relationship type."""
        if polarity > 1.5: return "strong friendship"
        if polarity > 0.5: return "friendship"
        if polarity < -1.5: return "strong enmity"
        if polarity < -0.5: return "enmity"
        return "neutral"

    def generate_report(self, relations: List[tuple], output_path: str):
        """Generate a text report of relationships."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Character Relationship Report\n")
                f.write("="*50 + "\n\n")
                
                for rel in sorted(relations, key=lambda x: x[2], reverse=True):
                    src, tgt, weight, typ, examples = rel
                    f.write(f"{src} → {tgt}\n")
                    f.write(f"Type: {typ}\n")
                    f.write(f"Interaction count: {weight}\n")
                    f.write(f"Examples:\n{examples}\n")
                    f.write("-"*50 + "\n")
                    
            logger.info(f"Report generated: {output_path}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")

    def visualize_network(self, graph: nx.Graph, output_html: str):
        """Generate interactive network visualization."""
        try:
            net = Network(
                notebook=False,
                height="800px",
                width="100%",
                bgcolor="#ffffff",
                font_color="black"
            )
            
            # Configure options
            net.set_options("""
            {
                "nodes": {
                    "scaling": {
                        "min": 10,
                        "max": 50
                    }
                },
                "physics": {
                    "stabilization": {
                        "enabled": true,
                        "iterations": 1000
                    }
                }
            }
            """)
            
            # Add nodes
            importance = nx.degree_centrality(graph)
            for node, data in graph.nodes(data=True):
                net.add_node(
                    node,
                    label=node,
                    size=importance[node] * 30 + 10,
                    title=f"""
                    Character: {node}
                    Aliases: {data.get('aliases', 'N/A')}
                    Importance: {importance[node]:.2f}
                    """
                )
            
            # Add edges
            for u, v, data in graph.edges(data=True):
                color = ("green" if data.get('type', '').startswith('friend') else
                         "red" if data.get('type', '').startswith('enmity') else
                         "gray")
                net.add_edge(
                    u, v,
                    value=data.get('weight', 1),
                    color=color,
                    title=f"""
                    Relation: {data.get('type', 'unknown')}
                    Interactions: {data.get('weight', 0)}
                    Polarity: {data.get('polarity', 0):.2f}
                    """
                )
            
            # Save visualization
            try:
                net.save_graph(output_html)
                logger.info(f"Visualization saved: {output_html}")
            except Exception as e:
                logger.error(f"Error saving HTML: {e}")
                # Fallback method
                html = net.generate_html()
                with open(output_html, 'w', encoding='utf-8') as f:
                    f.write(html)
                logger.info(f"Visualization regenerated: {output_html}")
                
        except Exception as e:
            logger.error(f"Error creating network: {e}")
            raise

    def process_corpus(self, folder_path: str, output_prefix: str = "network"):
        """Process a corpus of documents."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"Directory not found: {folder_path}")
        
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            raise ValueError("No PDF files found in directory")
        
        combined_graph = nx.Graph()
        all_relations = []
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                file_path = os.path.join(folder_path, pdf_file)
                text = self.extract_text_from_pdf(file_path)
                if not text:
                    continue
                    
                characters = self.extract_characters(text)
                graph, relations = self.detect_interactions(text, characters)
                
                combined_graph = nx.compose(combined_graph, graph)
                all_relations.extend(relations)
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
        
        # Generate outputs
        output_html = f"{output_prefix}.html"
        output_report = f"{output_prefix}_report.txt"
        
        self.generate_report(all_relations, output_report)
        self.visualize_network(combined_graph, output_html)
        
        return combined_graph


def main():
    """Main entry point."""
    try:
        # Configuration
        corpus_path = r"C:\Users\user\Desktop\M1ILSEN\AmsProjet3\Corpus_ASIMOV"
        output_prefix = "ASIMOV_network"
        
        # Initialize analyzer
        analyzer = CharacterInteractionAnalyzer(
            nlp_model="fr_core_news_lg",
            name_match_threshold=85,
            interaction_logging=False
        )
        
        # Process corpus
        analyzer.process_corpus(
            folder_path=corpus_path,
            output_prefix=output_prefix
        )
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Check requirements
    try:
        import spacy
        import networkx
        import pyvis
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install with: pip install spacy networkx pyvis python-louvain PyPDF2 fuzzywuzzy unidecode tqdm")
        sys.exit(1)
    
    # Download French model if needed
    try:
        import spacy
        spacy.load("fr_core_news_lg")
    except OSError:
        print("French language model not found. Downloading...")
        from spacy.cli import download
        download("fr_core_news_lg")
    
    main()