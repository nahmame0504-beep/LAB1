# Lab 1 — Prétraitement NLP en Python

Ce projet met en œuvre un pipeline modulaire de prétraitement de texte en Python pour le Traitement Automatique du Langage Naturel (NLP).

##  Objectif du projet

L'objectif est de transformer un texte brut en tokens exploitables pour des tâches d'analyse de données ou de Machine Learning à travers les étapes suivantes :
1. **Chargement & Normalisation :** Lecture du fichier UTF-8 et nettoyage des espaces superflus.
2. **Segmentation en phrases :** Découpage du texte avec `sent_tokenize`.
3. **Tokenisation en mots :** Découpage avec `word_tokenize`.
4. **Filtrage des Stop Words :** Suppression des mots vides avec `nltk.corpus.stopwords`.
5. **Suppression de la ponctuation :** Nettoyage des symboles et caractères typographiques.
6. **Normalisation de la casse :** Passage en minuscules tout en préservant certains acronymes/mots clés (`NLP`, `DATA`).
7. **Réduction morphologique :** Lemmatisation avec `WordNetLemmatizer`.

##  Arborescence du projet

```text
tp/
├── data/
│   └── corpus_en.txt

├── nlp_preprocess.py
└── README.md
```
<img width="1092" height="171" alt="LAB1" src="https://github.com/user-attachments/assets/fee26cf9-c5ed-4073-91ed-8a1d96418186" />



