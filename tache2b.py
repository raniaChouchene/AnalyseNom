import spacy
from collections import Counter, defaultdict
from itertools import islice
nlp = spacy.load("fr_core_news_sm")

fichier = r'C:\Users\user\Desktop\AmsProjet3\clearTextFonadionSample.txt'

with open(fichier, 'r', encoding='utf-8') as file:
    texte = file.read()

doc = nlp(texte)

mots = [token.text for token in doc if token.is_alpha]


unigrammes = Counter(mots)


bigrams = zip(mots, islice(mots, 1, None))
bigrammes = Counter(bigrams)

print("Table des Unigrammes :")
for mot, count in unigrammes.most_common(10): 
    print(f"{mot}: {count}")

print("\nTable des Bigrammes :")
for bigram, count in bigrammes.most_common(10): 
    print(f"{bigram}: {count}")
