# 📘 Classification Multiclasse des Caractères Tifinagh

> **Master IMSD  – Université Ibno Zohr  – Pr. M. Benaddy**

Ce projet implémente un **réseau de neurones multiclasses** pour classer les caractères Tifinagh issus de la base de données AMHCD. Il utilise un **Perceptron Multi-Couches (MLP)** avec deux couches cachées (64 et 32 neurones), activation ReLU et sortie Softmax.

## 📦 Contenu du Dépôt

- `data/` : Base de données AMHCD (images et CSV)
- `src/` : Scripts Python pour le modèle et l'entraînement
- `notebooks/` : Notebook Jupyter pour l’exploration
- `reports/` : Rapport final et figures
- `requirements.txt` : Liste des dépendances
- `README.md` : Ce fichier

## 📥 Téléchargement de la Base de Données

🔗 [Télécharger AMHCD sur Kaggle](https://www.kaggle.com/datasets/benaddym/amazigh-handwritten-character-database-amhcd) 

Placez les images dans :
data/tifinagh-images/
Pour entraîner le modèle :
python src/train.py

Résultats
Le modèle atteint une précision de ~88% sur le test set après 100 époques avec Adam et régularisation L2.