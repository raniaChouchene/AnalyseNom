import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv('my_submission.csv')  # Remplacez par le chemin vers votre fichier CSV

# Afficher les colonnes pour vérifier les noms exacts
print("Colonnes du fichier CSV : ", df.columns)

# Si nécessaire, nettoyez les noms des colonnes (en cas d'espaces ou caractères invisibles)
df.columns = df.columns.str.strip()

# Créer un graphe vide
G = nx.Graph()

# Ajouter des liens entre les personnages à partir du CSV
for index, row in df.iterrows():
    try:
        # Remplacez 'Character1' et 'Character2' par les noms des colonnes réelles si nécessaire
        G.add_edge(row['Character1'], row['Character2'])
    except KeyError:
        print("Erreur de clé, vérifier les noms des colonnes dans le CSV.")

# Visualiser le graphe
plt.figure(figsize=(10, 8))  # Ajustez la taille de la figure selon vos besoins
pos = nx.spring_layout(G)  # Disposition du graphe
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold', edge_color='gray')

# Afficher le graphe
plt.title("Visualisation des liens entre personnages")
plt.show()

# Sauvegarder l'image du graphe
plt.savefig('graph_image.png')  # Sauvegarde de l'image du graphe dans le fichier 'graph_image.png'
print("Le graphe a été sauvegardé sous 'graph_image.png'.")
