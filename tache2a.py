import spacy
from collections import Counter
from itertools import islice

nlp = spacy.load("fr_core_news_sm")

# Fonction pour extraire des bigrammes
def get_bigrams(words):
    return zip(words, islice(words, 1, None))

fichier = r'C:\Users\user\Desktop\AmsProjet3\clearTextFonadionSample.txt'

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
