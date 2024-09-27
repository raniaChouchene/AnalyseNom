import spacy
from collections import Counter
from itertools import islice

# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")

# Fonction pour extraire des bigrammes
def get_bigrams(words):
    return zip(words, islice(words, 1, None))

# Chemin du fichier texte
fichier = r'C:\Users\user\Desktop\AmsProjet3\clearTextFonadionSample.txt'
# Lire le contenu du fichier
with open(fichier, 'r', encoding='utf-8') as file:
    texte = file.read()

# Traiter le texte avec spaCy
doc = nlp(texte)

# Extraction de phrases
phrases = [sent.text for sent in doc.sents]

# Extraction des mots
mots = [token.text for token in doc if token.is_alpha]

# Comptage du nombre de caractères, de mots, de phrases
nb_caracteres = len(texte)
nb_mots = len(mots)
nb_phrases = len(phrases)

# Bigrammes
# bigrams = list(get_bigrams(mots))

# Fréquence des mots
freq_mots = Counter(mots)

# Résultats
print(f"Nombre de caractères: {nb_caracteres}")
print(f"Nombre de mots: {nb_mots}")
print(f"Nombre de phrases: {nb_phrases}")
# print(f"Fréquence des mots: {freq_mots}")
# print(f"Bigrammes: {bigrams}")
