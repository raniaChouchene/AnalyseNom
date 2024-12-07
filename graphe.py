import os
from itertools import combinations
from collections import defaultdict
import networkx as nx
import pandas as pd
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
from unidecode import unidecode
import spacy
from xml.sax.saxutils import escape

# Load spaCy's French model
nlp = spacy.load("fr_core_news_md")

def sanitize_node_id(node_id):
    """
    Sanitize node IDs to ensure they are valid for GraphML.
    - Replace spaces with underscores.
    - Strip leading and trailing whitespace.
    """
    return node_id.strip().replace(" ", "_")

def detect_and_draw_interactions(folder_path, output_csv="submission.csv", interaction_threshold=1):
    """
    Detect character interactions in text files, generate graphs for each chapter, and export results to a CSV.

    :param folder_path: Path to the folder containing subfolders with books and chapters.
    :param output_csv: Path to save the CSV results.
    :param interaction_threshold: Minimum number of interactions to include in the graph and CSV.
    """
    books = [
        (list(range(1, 20)), "paf"),
        (list(range(1, 19)), "lca"),
    ]

    df_dict = {"ID": [], "graphml": []}
    lieuxASupprimer = []
    listeDePersonnagesCorpus = []

    for chapters, book_code in books:
        for chapter in chapters:
            repertory = "prelude_a_fondation" if book_code == "paf" else "les_cavernes_d_acier"
            chapter_file = f"{folder_path}/{repertory}/chapter_{chapter}.txt.preprocessed"

            if not os.path.isfile(chapter_file):
                print(f"File not found: {chapter_file}")
                continue

            with open(chapter_file, "r", encoding="utf-8") as file:
                text = file.read()

            doc = nlp(text)

            # Filter entities and identify valid characters
            listePersonnages, listeLieux = [], []
            for ent in doc.ents:
                if ent.label_ == "PER" and len(ent.text) > 2:
                    ent_text = unidecode(ent.text.strip())
                    listePersonnages.append(ent_text)
                    listeDePersonnagesCorpus.append(ent_text)
                elif ent.label_ == "LOC":
                    listeLieux.append(ent.text.strip())

            # Filter out locations mistakenly detected as characters
            listePersonnagesTrier = set(listePersonnages)
            listeLieuxTrier = set(listeLieux)

            for personnage in listePersonnagesTrier:
                if listeLieux.count(personnage) > listePersonnages.count(personnage):
                    lieuxASupprimer.append(personnage)

            listePersonnages = [p for p in listePersonnagesTrier if p not in lieuxASupprimer]

            # Identify and group name variants
            listeNomsPersonnages = []
            for nom in listePersonnages:
                variantesNoms = [nom]
                for autreNom in listePersonnages:
                    if autreNom != nom and (autreNom in nom or nom in autreNom):
                        variantesNoms.append(autreNom)
                if not any(sorted(variantesNoms) == sorted(existing) for existing in listeNomsPersonnages):
                    listeNomsPersonnages.append(variantesNoms)

            # Create the interaction graph
            G = nx.Graph()

            # Train gensim LSI model on the chapter
            processed_docs = [simple_preprocess(doc) for doc in text.split("\n")]
            dictionary = corpora.Dictionary(processed_docs)
            corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
            lsi = models.LsiModel(corpus, num_topics=200)
            index = similarities.MatrixSimilarity(lsi[corpus])

            relations = defaultdict(list)
            for i in listeNomsPersonnages:
                for j in i:
                    for k in listeNomsPersonnages:
                        for l in k:
                            if j != l:
                                query_bow_j = dictionary.doc2bow(simple_preprocess(j))
                                query_lsi_j = lsi[query_bow_j]
                                sims_j = index[query_lsi_j]

                                query_bow_l = dictionary.doc2bow(simple_preprocess(l))
                                query_lsi_l = lsi[query_bow_l]
                                sims_l = index[query_lsi_l]

                                if any(sim_j > 0.09 and sim_l > 0.09 for sim_j, sim_l in zip(sims_j, sims_l)):
                                    relations[j].append(l)

            # Populate the graph with nodes and edges
            if relations:
                for source, targets in relations.items():
                    for target in targets:
                        sanitized_source = sanitize_node_id(source)
                        sanitized_target = sanitize_node_id(target)
                        G.add_edge(sanitized_source, sanitized_target)
            
            for group in listeNomsPersonnages:
                group = list(group)
                first_element = sanitize_node_id(group[0])
                remaining_elements = ";".join(sanitize_node_id(p) for p in group[1:])
                if first_element not in G.nodes:
                    G.add_node(first_element)
                G.nodes[first_element]["names"] = escape(
                    f"{first_element};{remaining_elements}" if remaining_elements else first_element
                )

            for node in G.nodes:
                if "names" not in G.nodes[node]:
                    G.nodes[node]["names"] = node
            
            
            df_dict["ID"].append("{}{}".format(book_code, chapter - 1))

            graphml = "".join(nx.generate_graphml(G))
            df_dict["graphml"].append(graphml)

    # Save results to CSV
    if len(df_dict["ID"]) != sum(len(chapters) for chapters, _ in books):
        print(f"Error: Chapter count mismatch. Expected {sum(len(chapters) for chapters, _ in books)}, found {len(df_dict['ID'])}.")
        return

    df = pd.DataFrame(df_dict)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Submission completed. CSV saved to {output_csv}")

# Define folder path
folder_path = r"C:\Users\user\Desktop\M1ILSEN\AmsProjet3\allfiles\reseaux-de-personnages-de-fondation-session-2"
detect_and_draw_interactions(folder_path)
