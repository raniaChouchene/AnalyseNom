import spacy
# pip install spacy
#python -m spacy download fr_core_news_sm

nlp = spacy.load("fr_core_news_sm")

def pos_tagging(input_file, output_file):
    # Lire le fichier d'entrée
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Analyse du texte avec Spacy
    doc = nlp(text)

    # Ouvrir le fichier de sortie
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in doc.sents:  # Traitement phrase par phrase
            words_pos = []
            for token in sent:
                # Filtrer la ponctuation ou autres symboles si nécessaire
                if token.is_punct:
                    continue
                # Formater le mot avec son étiquette de partie du discours
                words_pos.append(f"{token.text}/{token.pos_}")
            
            # Écrire la phrase analysée dans le fichier de sortie
            f.write(' '.join(words_pos) + '\n')

input_file = r'C:\Users\user\Desktop\AmsProjet3\clearTextFonadionSample.txt'
output_file = r'C:\Users\user\Desktop\AmsProjet3\output.txt'


# Appel de la fonction pour traiter le fichier
pos_tagging(input_file, output_file)

print(f"Analyse POS terminée. Résultats enregistrés dans {output_file}")
