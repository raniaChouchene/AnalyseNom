import re
import PyPDF2

def nettoyer_texte(texte):
    # Suppression des caractères spéciaux et des espaces en trop
    texte_propre = re.sub(r'\s+', ' ', texte)  # Remplace plusieurs espaces par un seul
    texte_propre = re.sub(r'[^\w\s]', '', texte_propre)  # Supprime tous les caractères spéciaux sauf les lettres et les chiffres
    return texte_propre.strip()

def extraire_texte_du_pdf(pdf_path):
    with open(pdf_path, 'rb') as fichier:
        lecteur_pdf = PyPDF2.PdfReader(fichier)
        texte_complet = ""
        for page_num in range(len(lecteur_pdf.pages)):
            page = lecteur_pdf.pages[page_num]
            texte_complet += page.extract_text()
        return texte_complet

def nettoyer_pdf(pdf_path, output_path):
    # Extraction du texte brut du PDF
    texte_brut = extraire_texte_du_pdf(pdf_path)
    
    # Nettoyage du texte
    texte_propre = nettoyer_texte(texte_brut)
    
    # Sauvegarde du texte nettoyé dans un fichier
    with open(output_path, 'w', encoding='utf-8') as fichier_sortie:
        fichier_sortie.write(texte_propre)

# Exemple d'utilisation
pdf_file = r'C:\Users\user\Desktop\AmsProjet3\Corpus_ASIMOV\Seconde_Fondation_sample.pdf'
output_file = r'C:\Users\user\Desktop\AmsProjet3\cleanTextSecondeFondation.txt'


nettoyer_pdf(pdf_file, output_file)

print("Nettoyage terminé. Le texte nettoyé est sauvegardé dans :", output_file)
