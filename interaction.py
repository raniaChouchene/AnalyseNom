from collections import defaultdict
import spacy
import os
from itertools import combinations

# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")

def detect_interactions_between_characters_in_folder(folder_path, output_file="interactions_summary.txt"):
    """
    Détecte toutes les interactions entre personnages dans tous les fichiers texte d'un dossier.
    :param folder_path: Chemin vers le dossier contenant les fichiers texte d'entrée.
    :param output_file: Chemin vers le fichier texte de sortie.
    """
    # Dictionnaire pour stocker les interactions entre personnages
    interactions = defaultdict(lambda: defaultdict(int))

    # Fonction pour vérifier si une entité est un personnage valide
    def is_valid_character(entity):
        # Exclure les entités non pertinentes comme des phrases incomplètes ou des mots fonctionnels
        invalid_phrases = ["supposez", "j’ai eu l", "et", "ou", "dans", "le", "la", "les", "de", "à", "pour", "avec", "sans", "qui", "quoi"]
        if any(phrase in entity.lower() for phrase in invalid_phrases):
            return False
        # Vérifie qu'il y a un prénom et un nom (et pas une phrase complète)
        if " " not in entity or len(entity.split()) < 2:
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

            # Extraction des entités nommées de type "PER" (Personnes)
            characters_in_text = [ent.text for ent in doc.ents if ent.label_ == "PER" and is_valid_character(ent.text)]

            # Imprimer les personnages extraits pour vérifier
            print(f"Personnages extraits du fichier {filename}: {characters_in_text}")

            # Si plusieurs personnages sont détectés dans le texte entier
            for char1, char2 in combinations(set(characters_in_text), 2):
                interactions[char1][char2] += 1
                interactions[char2][char1] += 1

    # Sauvegarder les interactions dans le fichier de sortie
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("\n--- Interactions entre personnages ---\n")
        for char1, related_chars in interactions.items():
            for char2, count in related_chars.items():
                if char1 < char2:  # pour éviter les doublons
                    file.write(f"{char1} <-> {char2}: {count} interactions\n")

    print(f"Extraction des interactions terminée. Résultats ajoutés dans {output_file}")

# Exemple d'utilisation
folder_path = r"C:\Users\ADMIN\Desktop\AMS_PROJET\kaggle"
detect_interactions_between_characters_in_folder(folder_path)
