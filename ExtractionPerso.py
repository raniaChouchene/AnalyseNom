import spacy
from collections import defaultdict
from itertools import combinations
import os

# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")

def detect_dynamic_characters_in_folder(folder_path, output_file="characters_summary.txt"):
    """
    Détecte dynamiquement les noms de personnages dans tous les fichiers texte d'un dossier.
    Ajoute les résultats dans le même fichier et additionne les occurrences des personnages déjà présents.
    :param folder_path: Chemin vers le dossier contenant les fichiers texte d'entrée.
    :param output_file: Chemin vers le fichier texte de sortie.
    """
    # Initialiser un dictionnaire pour stocker les occurrences globales des personnages
    global_character_mentions = defaultdict(int)

    # Fonction pour vérifier si une entité est un potentiel personnage
    def is_valid_character(entity, context_sentence):
        # Doit contenir au moins un prénom et un nom
        if " " not in entity:
            return False
        # Éviter les entités extrêmement courtes
        if len(entity) < 3:
            return False
        # Contexte syntaxique : vérifier si l'entité agit ou interagit dans le texte
        action_verbs = [token for token in context_sentence if token.pos_ == "VERB"]
        if not action_verbs:
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

            # Stockage des mentions de personnages dans ce fichier
            character_mentions = defaultdict(int)

            # Extraction des entités nommées de type "PER" (Personnes)
            for ent in doc.ents:
                if ent.label_ == "PER":
                    # Valider avec le contexte de la phrase
                    if is_valid_character(ent.text, ent.sent):
                        character_mentions[ent.text] += 1

            # Ajouter les occurrences de ce fichier au total global
            for character, count in character_mentions.items():
                global_character_mentions[character] += count

    # Résolution des alias : Regrouper les noms similaires
    character_aliases = defaultdict(list)
    names = list(global_character_mentions.keys())

    for name1, name2 in combinations(names, 2):
        # Si les noms partagent le même nom de famille
        if name1.split()[-1] == name2.split()[-1]:
            character_aliases[name1].append(name2)
            character_aliases[name2].append(name1)

    # Consolidation des occurrences
    consolidated_mentions = defaultdict(int)
    processed = set()

    for character, aliases in character_aliases.items():
        if character not in processed:
            total_count = global_character_mentions[character] + sum(global_character_mentions[alias] for alias in aliases)
            consolidated_mentions[character] = total_count
            processed.update([character] + aliases)

    # Ajouter les personnages sans alias consolidé
    for character in names:
        if character not in consolidated_mentions:
            consolidated_mentions[character] = global_character_mentions[character]

    # Sauvegarder les résultats dans le fichier de sortie
    with open(output_file, 'a', encoding='utf-8') as file:  # Mode append pour ajouter sans écraser
        file.write("\n--- Résultats extraits des fichiers ---\n")
        for character, count in sorted(consolidated_mentions.items(), key=lambda x: x[1], reverse=True):
            file.write(f"{character}: {count} occurrences\n")

    print(f"Extraction terminée. Résultats ajoutés dans {output_file}")


# Exemple d'utilisation
folder_path = r"C:\Users\user\Desktop\M1ILSEN\AmsProjet3\DataFile"
detect_dynamic_characters_in_folder(folder_path)
