import re
import PyPDF2

def nettoyer_texte(texte):
    # Remplacement des multiples espaces par un seul, mais garder les retours à la ligne
    texte_propre = re.sub(r'[^\S\r\n]+', ' ', texte) 
    texte_propre = re.sub(r'[^\w\s\.\,\n]', '', texte_propre)  
    
    # Supprimer les espaces avant les ponctuations et s'assurer d'un espace après chaque point ou virgule
    texte_propre = re.sub(r'\s+([.,])', r'\1', texte_propre)  
    texte_propre = re.sub(r'([.,])(\S)', r'\1 \2', texte_propre)  
    texte_propre = re.sub(r'\.\s+', '.\n', texte_propre)
    
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
    texte_brut = extraire_texte_du_pdf(pdf_path)
    texte_propre = nettoyer_texte(texte_brut)

    with open(output_path, 'w', encoding='utf-8') as fichier_sortie:
        fichier_sortie.write(texte_propre)


pdf_file = r'C:\Users\user\Desktop\AmsProjet3\Corpus_ASIMOV\Fondation_sample.pdf'
output_file = r'C:\Users\user\Desktop\AmsProjet3\clearTextFonadionSample.txt'

nettoyer_pdf(pdf_file, output_file)

print("Nettoyage terminé. Le texte nettoyé est sauvegardé dans :", output_file)
