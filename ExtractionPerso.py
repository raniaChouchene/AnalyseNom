import spacy
from collections import defaultdict
from itertools import combinations

# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")

def detect_dynamic_characters(input_file, output_file="characters_summary.txt"):
    """
    Détecte dynamiquement les noms de personnages dans un fichier texte.
    :param input_file: Chemin vers le fichier texte d'entrée.
    :param output_file: Chemin vers le fichier texte de sortie.
    """
    # Lire le contenu du fichier d'entrée
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Analyse du texte avec spaCy
    doc = nlp(text)

    # Stockage des mentions de personnages
    character_mentions = defaultdict(int)

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

    # Extraction des entités nommées de type "PER" (Personnes)
    for ent in doc.ents:
        if ent.label_ == "PER":
            # Valider avec le contexte de la phrase
            if is_valid_character(ent.text, ent.sent):
                character_mentions[ent.text] += 1

    # Résolution des alias : Regrouper les noms similaires
    character_aliases = defaultdict(list)
    names = list(character_mentions.keys())

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
            total_count = character_mentions[character] + sum(character_mentions[alias] for alias in aliases)
            consolidated_mentions[character] = total_count
            processed.update([character] + aliases)

    # Ajouter les personnages sans alias consolidé
    for character in names:
        if character not in consolidated_mentions:
            consolidated_mentions[character] = character_mentions[character]

    # Sauvegarder les résultats dans un fichier de sortie
    with open(output_file, 'w', encoding='utf-8') as file:
        for character, count in sorted(consolidated_mentions.items(), key=lambda x: x[1], reverse=True):
            file.write(f"{character}: {count} occurrences\n")

    print(f"Extraction des personnages terminée. Résultats sauvegardés dans {output_file}")


# Exemple d'utilisation
input_file = r"C:\Users\user\Desktop\M1ILSEN\AmsProjet3\DataFile\chapter_3.txt"
detect_dynamic_characters(input_file)
