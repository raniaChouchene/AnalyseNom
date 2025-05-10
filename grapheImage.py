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
from typing import Dict, List, Set, Tuple, Optional 

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
        logger.info(f"Loaded {len(self.anti_dic)} terms in anti-dictionary")

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
        """Load anti-dictionary with improved error handling and multiple encoding attempts."""
        anti_dic_set = set()
        try:
            if os.path.exists(path):
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        with open(path, 'r', encoding=encoding) as f:
                            anti_dic_set = {unidecode(line.strip().lower()) for line in f if line.strip()}
                        logger.info(f"Successfully loaded anti-dictionary with {len(anti_dic_set)} entries using {encoding}")
                        break
                    except UnicodeDecodeError:
                        logger.debug(f"Failed to decode using {encoding}, trying next encoding")
                    except Exception as e:
                        logger.warning(f"Error reading file with {encoding}: {e}")
                
                if not anti_dic_set:
                    logger.warning("Could not load anti-dictionary with any encoding")
            else:
                logger.warning(f"Anti-dictionary file not found: {path}")
        except Exception as e:
            logger.warning(f"Could not load anti-dictionary: {e}")
        
        return anti_dic_set

    def _init_polarity_dictionaries(self):
        """Initialize polarity dictionaries."""
        self.friendship_verbs = {
            "aimer", "adorer", "soutenir", "aider", "protéger", "défendre",
            "sauver", "réconforter", "féliciter", "encourager", "apprécier",
            "embrasser", "câliner", "sourire", "complimenter", "secourir",
            "respecter", "admirer", "accompagner", "assister", "consoler",
            "rassurer", "chérir", "vénérer", "estimer", "honorer", "louer",
            "sympathiser", "collaborer", "coopérer", "s'entendre", "s'accorder",
            "réconcilier", "pardonner", "applaudir", "remercier", "gratifier",
            "valoriser", "caresser", "soigner", "guérir", "servir", "plaire"
        }
        
        # Expanded enmity verbs
        self.enmity_verbs = {
            "détester", "haïr", "combattre", "trahir", "attaquer", "blesser",
            "tuer", "frapper", "insulter", "menacer", "voler", "tromper",
            "humilier", "piéger", "jalouser", "poignarder", "dénoncer",
            "mépriser", "abhorrer", "exécrer", "abominer", "abuser", "accuser",
            "agresser", "aliéner", "anéantir", "anathématiser", "antagoniser",
            "asservir", "bafouer", "bannir", "battre", "calomnier", "condamner",
            "contrarier", "contredire", "corrompre", "critiquer", "dénigrer",
            "désavouer", "déshonorer", "diffamer", "discréditer", "disqualifier",
            "dominer", "éliminer", "empoisonner", "emprisonner", "enfermer",
            "enlever", "enrager", "escroquer", "exclure", "exploiter", "exterminer",
            "harceler", "maltraiter", "marginaliser", "médire", "mentir",
            "molester", "moquer", "mortifier", "nuire", "offenser", "opprimer",
            "persécuter", "ridiculiser", "saboter", "torturer", "tyranniser", "analyser", "calculer", "prédire", "projeter", "planifier", "théoriser",
            "négocier", "converser", "transmettre", "communiquer", "voyager"
        }
        
        # Expanded neutral verbs
        self.neutral_verbs = {
            "parler", "voir", "rencontrer", "observer", "écouter", "dire",
            "demander", "répondre", "regarder", "trouver", "penser", "croire",
            "entendre", "apercevoir", "remarquer", "découvrir", "visiter",
            "interroger", "questionner", "informer", "annoncer", "consulter",
            "contempler", "contacter", "connaître", "considérer", "discuter",
            "examiner", "expliquer", "fixer", "identifier", "indiquer", "mentionner",
            "percevoir", "reconnaître", "repérer", "signaler", "suivre", "surveiller",
            "téléphoner", "vérifier", "étudier", "comprendre", "sentir"
        }
        
        # Expanded friendship adjectives
        self.friendship_adjectives = {
            "gentil", "sympathique", "bienveillant", "loyal", "aimable",
            "généreux", "honnête", "fidèle", "chaleureux", "dévoué", "courageux",
            "affectueux", "amical", "attentionné", "bon", "compatissant", "compréhensif",
            "confiant", "courtois", "doux", "empathique", "encourageant", "fiable",
            "indulgent", "juste", "pacifique", "patient", "protecteur", "rassurant",
            "reconnaissant", "respectueux", "sensible", "serviable", "sincère",
            "tendre", "tolérant", "vertueux", "agréable", "charmant", "plaisant",
            "séduisant", "attrayant", "envoûtant", "fascinant", "irrésistible","brillant", "intelligent", "sage", "savant", "visionnaire", "clairvoyant",
            "logique", "rationnel", "pragmatique", "stratégique", "diplomate"
        }
        
        # Expanded enmity adjectives
        self.enmity_adjectives = {
            "méchant", "hostile", "dangereux", "traître", "cruel",
            "égoïste", "fourbe", "malveillant", "violent", "haineux", "sadique",
            "abominable", "agressif", "amer", "antipathique", "arrogant", "brutal",
            "cynique", "déloyal", "dépravé", "détestable", "diabolique", "dur",
            "envieux", "féroce", "froid", "furieux", "hypocrite", "impitoyable",
            "infâme", "injuste", "insolent", "insupportable", "jaloux", "machiavélique",
            "malhonnête", "mauvais", "menaçant", "mesquin", "monstrueux", "narcissique",
            "odieux", "offensant", "pathétique", "perfide", "pernicieux", "pervers",
            "provocateur", "répugnant", "sournois", "tyrannique", "vengeur", "vicieux","impérial", "dictatorial", "oppressif", "totalitaire", "manipulateur",
            "psychopathe", "corrompu", "destructeur", "opportuniste", "subversif", "renégat", "infidèle", "usurpateur", "calculateur", "toxique","intrigant", "dominant", "abusif", "diviseur", "séparatiste","faux", "dissimulé", "instable", "imprévisible", "autoritaire"
        }


    def is_valid_character_name(self, name: str) -> bool:
        """Check if a name is valid for a character with improved anti-dictionary checking."""
        if not name or len(name) < self.min_name_length:
            return False
        
        name_lower = unidecode(name.lower())
        if any(char.isdigit() for char in name):
            return False
            
        # Exclude generic terms
        generic_terms = {"monsieur", "madame", "docteur", "professeur", "capitaine", "s absolument", "s", "absolument" , "vii", "ii", "messieurs", "poli","salvor", "excellence","gente dame","gentedame"}
        
        # Check if any word in the name appears in anti_dic
        name_words = name_lower.split()
        if any(word in self.anti_dic for word in name_words):
            return False
            
        first_word = name_words[0] if name_words else ""
        if first_word in generic_terms:
            return False
        
        # Check if the full name is in anti_dic    
        return name_lower not in self.anti_dic

    def normalize_name(self, name: str) -> str:
        """Normalize character names."""
        # Remove titles and punctuation
        name = re.sub(r"(^|\s)(M|Mme|Dr|Pr)\.?\s+", " ", name, flags=re.IGNORECASE)
        name = re.sub(r"[^\w\s]", "", name)
        return unidecode(name.strip().replace("_", " ").lower())

    def advanced_name_matching(self, names: List[str]) -> Dict[str, List[str]]:
        """Group similar character names with improved anti-dictionary filtering."""
        # First normalize all names
        normalized_names = [self.normalize_name(name) for name in names]
        
        # Thorough anti-dictionary check: exclude any name that contains words from anti_dic
        valid_names = []
        for name in normalized_names:
            name_parts = name.split()
            if name and not any(part in self.anti_dic for part in name_parts) and name not in self.anti_dic:
                valid_names.append(name)
        
        unique_names = list(set(valid_names))
        
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
            elif name not in processed:  # Include singletons
                grouped[name] = [name]
                processed.add(name)

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
        """Extract characters from text using NER with improved anti-dictionary filtering."""
        doc = self.nlp(text)
        characters = []
        
        for ent in doc.ents:
            if ent.label_ == "PER":
                name = ent.text.strip()
                # Initial validity check
                if self.is_valid_character_name(name):
                    characters.append(name)
        
        # Double-check no anti-dictionary terms slipped through
        filtered_characters = []
        for name in characters:
            normalized = self.normalize_name(name)
            name_parts = normalized.split()
            
            # Skip if any part is in anti_dic or if the whole name is in anti_dic
            if any(part in self.anti_dic for part in name_parts) or normalized in self.anti_dic:
                logger.debug(f"Excluded character from anti-dictionary: {name}")
                continue
                
            filtered_characters.append(name)
        
        return self.advanced_name_matching(filtered_characters)

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
        """Detect character interactions with full anti-dictionary filtering."""
        G = nx.Graph()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        relations = []

        # Final anti-dictionary check before adding nodes
        filtered_characters = {}
        for canon, aliases in characters.items():
            # Check canonical name against anti-dictionary
            canon_parts = unidecode(canon.lower()).split()
            if canon not in self.anti_dic and not any(part in self.anti_dic for part in canon_parts):
                # Check all aliases against anti-dictionary
                valid_aliases = []
                for alias in aliases:
                    alias_parts = unidecode(alias.lower()).split()
                    if alias not in self.anti_dic and not any(part in self.anti_dic for part in alias_parts):
                        valid_aliases.append(alias)
                
                if valid_aliases:  # Only add if we have valid aliases
                    filtered_characters[canon] = valid_aliases

        # Add nodes
        for canon, aliases in filtered_characters.items():
            G.add_node(canon, aliases=", ".join(aliases), size=10)

        # Analyze interactions
        for src, src_aliases in filtered_characters.items():
            for tgt, tgt_aliases in filtered_characters.items():
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
                
                f.write(f"Total relationships found: {len(relations)}\n\n")
                
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
                    size=importance[node] * 50 + 15,
                    title=f"""
                    Character: {node}
                    Aliases: {data.get('aliases', 'N/A')}
                    Importance: {importance[node]:.2f}
                    """,
                    borderWidth=2, 
                    borderWidthSelected=3,
                    font={"size": 12}
                    
                )
            
            # Add edges
            for u, v, data in graph.edges(data=True):
                color = ("green" if data.get('type', '').startswith('friend') else
                         "red" if data.get('type', '').startswith('enmity') else
                         "gray")
                weight = data.get('weight', 1)
                net.add_edge(
                    u, v,
                    value=weight,
                    width=max(1, min(weight, 10)),
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
        all_characters = {}
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                file_path = os.path.join(folder_path, pdf_file)
                text = self.extract_text_from_pdf(file_path)
                if not text:
                    logger.warning(f"No text extracted from {pdf_file}")
                    continue
                    
                # Extract characters with anti-dictionary filtering
                characters = self.extract_characters(text)
                logger.info(f"Found {len(characters)} characters in {pdf_file} after filtering")
                
                # Check any anti-dictionary terms that might have slipped through
                filtered_chars = {}
                for canon, aliases in characters.items():
                    canon_lower = unidecode(canon.lower())
                    
                    # Skip if canonical name is in anti-dictionary
                    if canon_lower in self.anti_dic or any(part in self.anti_dic for part in canon_lower.split()):
                        logger.debug(f"Excluded character from anti-dictionary: {canon}")
                        continue
                    
                    # Filter aliases against anti-dictionary
                    valid_aliases = []
                    for alias in aliases:
                        alias_lower = unidecode(alias.lower())
                        if alias_lower not in self.anti_dic and not any(part in self.anti_dic for part in alias_lower.split()):
                            valid_aliases.append(alias)
                    
                    if valid_aliases:  # Only add if we have valid aliases
                        filtered_chars[canon] = valid_aliases
                
                # Update the character list
                all_characters.update(filtered_chars)
                
                # Detect interactions
                graph, relations = self.detect_interactions(text, filtered_chars)
                
                # Combine with existing graph
                combined_graph = nx.compose(combined_graph, graph)
                all_relations.extend(relations)
                
                logger.info(f"Processed {pdf_file}: found {len(relations)} relationships")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
        
        # Generate outputs
        output_html = f"{output_prefix}.html"
        output_report = f"{output_prefix}_report.txt"
        output_characters = f"{output_prefix}_characters.txt"
        
        # Save character list
        try:
            with open(output_characters, 'w', encoding='utf-8') as f:
                f.write("Character List\n")
                f.write("="*50 + "\n\n")
                
                for canon, aliases in all_characters.items():
                    f.write(f"{canon}\n")
                    f.write(f"Aliases: {', '.join(aliases)}\n")
                    f.write("-"*50 + "\n")
            
            logger.info(f"Character list saved: {output_characters}")
        except Exception as e:
            logger.error(f"Error saving character list: {e}")
        
        # Generate report and visualization
        self.generate_report(all_relations, output_report)
        self.visualize_network(combined_graph, output_html)
        
        logger.info(f"Processing complete. Found {len(all_characters)} characters and {len(all_relations)} relationships.")
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
            interaction_logging=True,
            anti_dic_path=r"C:\Program Files\Python 3.12\antiDic.txt"
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