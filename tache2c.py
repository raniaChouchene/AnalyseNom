import spacy

# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")

# Fonction pour générer la version POS du texte
def generer_pos_texte(texte):
    # Traiter le texte avec spaCy
    doc = nlp(texte)

    # Créer une version avec les étiquettes POS
    pos_texte = []
    for token in doc:
        pos_texte.append(f"{token.text}/{token.pos_}")

    return " ".join(pos_texte)

# Chemin du fichier texte
fichier = r'C:\Users\ADMIN\Desktop\AMS_PROJET\tp3_reseau\generationPersonnage\cleanTextSecondeFondation.txt'

# Lire le contenu du fichier
with open(fichier, 'r', encoding='utf-8') as file:
    texte = file.read()

# Générer la version POS du texte
resultat = generer_pos_texte(texte)

# Afficher ou sauvegarder le résultat
print(resultat)
