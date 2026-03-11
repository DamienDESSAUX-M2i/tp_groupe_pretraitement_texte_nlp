# TP Groupe : Sentiment Analysis sur Amazon Reviews

**Équipe:** 3-4 personnes
**Livrables:** Notebook complet + Rapport d'analyse

---

## Objectif

Construire un système de classification de sentiments sur un corpus réel d'avis Amazon. Le projet intègre toutes les étapes du pipeline NLP : exploration, nettoyage, feature engineering, modélisation multi-modèles, validation et analyse approfondie des performances.

---

## Dataset Real : Amazon Reviews

**Source:** Kaggle - Amazon Reviews for Sentiment Analysis
- URL: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

**Caractéristiques:**
- 4 millions+ d'avis Amazon
- Classes: positif (⭐⭐⭐⭐⭐, ⭐⭐⭐⭐) et négatif (⭐⭐, ⭐)
- Textes bruts, non nettoyés
- Métadonnées: rating, date implicite
- Format: train.ft.txt et test.ft.txt

**Pour le TP (utiliser un subset):**
- Télécharger les données
- Prendre 5,000-10,000 avis pour temps raisonnable
- Stratifier par classe (équilibre 50/50)
- Garder le format original pour réalisme

**Alternative si Kaggle problématique:**
- Utilisez Movie Reviews Dataset (Pang & Lee)
- URL: https://www.cs.cornell.edu/people/pabo/movie-review-data/

---

## Contexte Réel

Vous travaillez pour une plateforme d'e-commerce. Objectifs business:
- Prédire automatiquement le sentiment des avis clients
- Filtrer les faux avis (sentiments incohérents)
- Identifier les produits problématiques automatiquement
- Accélérer la modération manuelle

Challenges réels du dataset:
- Textes très hétérogènes (courts et longs)
- Sarcasme et ironie présents
- Fautes d'orthographe et HTML
- Avis acheteur vs non-acheteur
- Avis vérifiés et non-vérifiés

---

## Répartition des Rôles

| Rôle | Responsabilités |
|------|------------------|
| **Lead Data** | Chargement, EDA, nettoyage avancé |
| **Lead ML** | Modélisation, tuning, comparaison |
| **Lead Analyse** | Métriques, visualisations, insights |
| **Lead Synthesis** | Rapport, documentation, conclusions |

---

## Phase 1 : Exploration & Analyse

Comprendre la structure réelle des données Amazon.

**Tâches:**

1. Charger le dataset
   - Parser le format .ft.txt
   - Extraire labels et textes
   - Vérifier intégrité

2. Statistiques descriptives
   - Nombre d'avis par classe
   - Longueur moyenne, min, max
   - Distribution des longueurs
   - Vocabulaire total unique

3. Analyse textuelle
   - Top 50 mots positifs et négatifs
   - Bigrammes et trigrammes intéressants
   - URLs, emails, mentions
   - Caractères spéciaux communs

4. Patterns réalistes
   - Sarcasme détectable? (exemples)
   - Avis trop courts (bruit?)
   - Avis trop longs (copier-coller?)
   - Répétitions suspectes

5. Visualisations
   - Distribution sentiments (class balance)
   - Longueur par sentiment
   - Word clouds positif/négatif
   - Caractéristiques textuelles

---

## Phase 2 : Prétraitement Réaliste

Nettoyer les données comme elles arrivent réellement.

**Tâches:**

1. Nettoyage brut
   - Suppression doublons exacts
   - Suppression avis < 5 caractères
   - Suppression avis avec > 95% caractères spéciaux
   - Encodage UTF-8 (gérer caractères non-ASCII)

2. HTML & Markup
   - Suppression tags HTML
   - Conversion entités (&amp; → &)
   - Suppression URLs
   - Suppression mentions utilisateur

3. Normalisation
   - Minuscules
   - Expansion contractions (don't → do not)
   - Suppression accents (optionnel)
   - Nombres : <NUM> ou supprimer

4. Tokenization
   - Word tokenization (NLTK)
   - Sentence tokenization si pertinent
   - Gestion ponctuation

5. Stopwords & Filtering
   - NLTK stopwords
   - Analyse impact avant/après
   - Considérer stopwords négatifs importants (not, no, don't)

6. Lemmatisation
   - spaCy pour lemmatisation
   - Vs stemming : comparer
   - Mesurer changement de dimension

---

## Phase 3 : Feature Engineering

Créer des représentations numériques du texte Amazon brut.

**Tâches:**

1. TF-IDF
   ```python
   TfidfVectorizer(
       max_features=5000,
       min_df=2,
       max_df=0.8,
       ngram_range=(1, 2),
       lowercase=True
   )
   ```

2. Word2Vec
   - Entraîner sur corpus complet
   - vector_size=300, window=5
   - Document embeddings (moyenne)
   - Analyser similarités

3. FastText (Alternative)
   - Gère out-of-vocabulary
   - Subword information
   - Comparer avec Word2Vec

4. Features additionnelles
   - Longueur texte (raw, log)
   - Nombre mots uniques
   - Ratio ponctuation
   - Nombre majuscules
   - Sentiment lexicon score (TextBlob)

5. Analyse feature
   - Quelle approche discrimine mieux?
   - Features corrélées?
   - Réduction dimension si besoin (PCA)

---

## Phase 4 : Modélisation Comparative

Tester différentes approches sur données réelles.

**Modèles à implémenter:**

1. **Logistic Regression** (Baseline)
   - Simple et interprétable
   - FastText + LR souvent bon en NLP
   - Paramètres: C, penalty, solver

2. **Support Vector Machine**
   - Kernel: linear pour TF-IDF
   - Paramètres: C, gamma
   - Bon avec données haute-dimension

3. **Random Forest**
   - Robuste aux outliers
   - Feature importance clairement identifiable
   - Paramètres: n_estimators, max_depth

4. **Ensemble Methods**
   - Voting (LR + SVM + RF)
   - Stacking avec meta-learner
   - Comparer impact de chaque modèle

5. **Optionnel: Naive Bayes**
   - Baseline très rapide
   - Bon baseline pour données texte
   - Multinomial vs Bernoulli

**Pipeline complet:**
```python
# Pour chaque modèle:
1. Vectoriser train/test
2. Scaler si nécessaire
3. GridSearchCV avec cv=5 (stratified)
4. Entraîner sur train
5. Évaluer sur test
6. Sauvegarder modèle
```

---

## Phase 5 : Validation Robuste

Évaluer la généralisation sur données réelles.

**Tâches:**

1. Cross-validation
   - Stratified K-Fold (k=5)
   - Scores par fold
   - Mesurer variance
   - Identifier folds difficiles

2. Métriques complètes
   - Accuracy, Precision, Recall, F1
   - ROC-AUC score
   - Confusion matrix
   - Classification report par classe

3. Test de robustesse
   - Ajouter typos aléatoires
   - Supprimer mots aléatoires
   - Tester sur distribution différente (si dispo)
   - Mesurer dégradation

4. Erreur analysis
   - Analyser 30 faux positifs
   - Analyser 30 faux négatifs
   - Patterns d'erreur?
   - Cas ambigus identifiés?

5. Comparaison final
   - Meilleur modèle sélectionné
   - Trade-offs justifiés
   - Performance acceptable?

---

## Phase 6 : Analyse Approfondie

Extraire insights métier du modèle.

**Tâches:**

1. Feature importance
   - Top 30 features positives
   - Top 30 features négatives
   - Palavras clés par sentiment
   - Comparaison modèles

2. Analyse erreurs
   - Qu'est-ce qui rend l'avis difficile?
   - Sarcasme non-détecté?
   - Mixte (positif+négatif)?
   - Langage peu commun?

3. Insights métier
   - Aspects critiques des avis
   - Patterns de satisfaction
   - Problèmes communs identifiés
   - Recommandations pour vendeurs

4. Fairness & Bias
   - Est-ce que modèle a des biais?
   - Performance homogène entre groupes?
   - Disparités à noter?

---

## Phase 7 : Rapport & Présentation

Documenter tout le travail.

**Rapport (structure markdown):**

```
1. EXECUTIVE SUMMARY
   - Challenge business
   - Approche
   - Résultats clés

2. DATASET & EXPLORATION
   - Source et caractéristiques
   - Distribution classes
   - Observations clés
   - Challenges identifiés

3. MÉTHODOLOGIE
   - Preprocessing pipeline
   - Features utilisées
   - Modèles testés
   - Stratégie de validation

4. RÉSULTATS QUANTIFIÉS
   - Tableau comparatif modèles
   - Métriques détaillées
   - Meilleur modèle justifié
   - Confusion matrix

5. ANALYSE APPROFONDIE
   - Feature importance
   - Erreurs typiques
   - Insights métier
   - Problèmes identifiés

6. LIMITATIONS HONNÊTES
   - Données insuffisantes pour quoi?
   - Biais potentiels
   - Cas non-couverts
   - Améliorations futures

7. CONCLUSIONS
   - Production-ready?
   - Recommandations court terme
   - Recommandations long terme
```

---

## Livrables Finaux

1. **Notebook Jupyter**
   - Exécutable et documenté
   - Code commenté
   - Résultats visibles

2. **Modèles**
   - Best model (.pkl)
   - Vectorizers (.pkl)
   - Scaler (.pkl)
   - Metadata (.json)

3. **Rapport**
   - Complet et bien structuré
   - Visualisations intégrées
   - Données chiffrées

4. **Données Préparées**
   - Train/test sets nettoyés
   - Features engineered
   - Documentation preprocessing

---

## Conseils Pratiques

- Commencez avec petit subset (1000 avis) pour iterations rapides
- Utilisez random_state=42 partout
- Documentez chaque décision de preprocessing
- Testez avec TF-IDF d'abord (simple, efficace)
- Git commits réguliers
- Pair programming sur décisions critiques
- Sauvegarder modèles intermédiaires

---

## Ressources

- scikit-learn: https://scikit-learn.org
- NLTK: https://www.nltk.org/
- Gensim (Word2Vec): https://radimrehurek.com/gensim/
- Kaggle Dataset: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

---

**Bonne chance!**
