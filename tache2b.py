import spacy
from collections import Counter, defaultdict
from itertools import islice
# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")

# Chemin du fichier texte
fichier = r'C:\Users\ADMIN\Desktop\AMS_PROJET\tp3_reseau\generationPersonnage\cleanTextSecondeFondation.txt'

# Lire le contenu du fichier
with open(fichier, 'r', encoding='utf-8') as file:
    texte = file.read()

# Traiter le texte avec spaCy
doc = nlp(texte)

# Extraction des mots
mots = [token.text for token in doc if token.is_alpha]

# Calcul des unigrammes
unigrammes = Counter(mots)

# Calcul des bigrammes
bigrams = zip(mots, islice(mots, 1, None))
bigrammes = Counter(bigrams)

# Affichage des résultats
print("Table des Unigrammes :")
for mot, count in unigrammes.most_common(10):  # Affiche les 10 premiers unigrammes
    print(f"{mot}: {count}")

print("\nTable des Bigrammes :")
for bigram, count in bigrammes.most_common(10):  # Affiche les 10 premiers bigrammes
    print(f"{bigram}: {count}")
