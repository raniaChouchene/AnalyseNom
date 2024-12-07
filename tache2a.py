import spacy
from collections import Counter
from itertools import islice

nlp = spacy.load("fr_core_news_sm")

# Fonction pour extraire des bigrammes
def get_bigrams(words):
    return zip(words, islice(words, 1, None))


# Chemin du fichier texte
fichier = r'C:\Users\ADMIN\Desktop\AMS_PROJET\tp3_reseau\generationPersonnage\cleanTextSecondeFondation.txt'
# Lire le contenu du fichier


with open(fichier, 'r', encoding='utf-8') as file:
    texte = file.read()

doc = nlp(texte)

phrases = [sent.text for sent in doc.sents]
mots = [token.text for token in doc if token.is_alpha]

nb_caracteres = len(texte)
nb_mots = len(mots)
nb_phrases = len(phrases)

freq_mots = Counter(mots)

print(f"Nombre de caractères: {nb_caracteres}")
print(f"Nombre de mots: {nb_mots}")
print(f"Nombre de phrases: {nb_phrases}")
