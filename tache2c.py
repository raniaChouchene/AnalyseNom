import spacy

# Charger le modèle français de spaCy
nlp = spacy.load("fr_core_news_sm")

# Fonction pour générer les étiquettes POS dans le texte
def generer_pos_texte(texte):
    doc = nlp(texte)
    pos_texte = [f"{token.text}/{token.pos_}" for token in doc if not token.is_punct]
    return " ".join(pos_texte)

# Fonction pour traiter un fichier et enregistrer l'analyse POS dans un fichier de sortie
def pos_tagging(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        texte = f.read()

    resultat = generer_pos_texte(texte)

    # Enregistrer dans le fichier de sortie
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(resultat)

    print(f"Analyse POS terminée. Résultats enregistrés dans {output_file}")


# Définition des chemins de fichiers
input_file = r'C:\Users\user\Desktop\AmsProjet3\prelude_a_fondation\chapter_15.txt.preprocessed'
output_file = r'C:\Users\user\Desktop\AmsProjet3\output2.txt'

# Appel de la fonction
pos_tagging(input_file, output_file)
