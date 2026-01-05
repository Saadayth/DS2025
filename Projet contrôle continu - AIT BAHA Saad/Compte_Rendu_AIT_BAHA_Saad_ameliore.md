---
Titre: "Rapport académique – Implémentation technique (Python)"
Soustitre: "Dataset UCI : Estimation of Obesity Levels Based on Eating Habits and Physical Condition (id=544)"
Institution: "École Nationale de Commerce et de Gestion (ENCG) – Settat"
Année: "2024–2025"
Étudiant: "AIT BAHA Saad"
"Groupe 1 – CAC"
---

# École Nationale de Commerce et de Gestion – Settat  
## Année universitaire 2024–2025  
### Rapport académique – Implémentation technique (Python)  
### Dataset UCI : *Estimation of Obesity Levels Based on Eating Habits and Physical Condition* (id=544)

**Étudiant :** AIT BAHA Saad  
**Groupe / Parcours :** Groupe 1 – CAC  

<div align="center">
  <img src="https://github.com/user-attachments/assets/1fc30205-0df3-4b78-82c5-246950d26dd6" alt="Photo / Illustration" width="180"/>
</div>

**Vidéo (démonstration / livrables) :**  
- https://drive.google.com/drive/folders/1aeCVIo1qphFiTGuFCcCOXDPMMw3JJ6Bw?usp=sharing

---

## Table des matières

1. [Introduction](#1-introduction)  
2. [Présentation du dataset](#2-présentation-du-dataset)  
   1. [Source et contexte](#21-source-et-contexte)  
   2. [Unité statistique et taille de l’échantillon](#22-unité-statistique-et-taille-de-léchantillon)  
   3. [Variable cible](#23-variable-cible)  
   4. [Familles de variables explicatives](#24-familles-de-variables-explicatives)  
3. [Méthodologie générale](#3-méthodologie-générale)  
   1. [Environnement technique](#31-environnement-technique)  
   2. [Pré-traitement des données](#32-pré-traitement-des-données)  
   3. [Feature engineering](#33-feature-engineering)  
   4. [Partitionnement des données](#34-partitionnement-des-données)  
4. [Analyse exploratoire des données (EDA)](#4-analyse-exploratoire-des-données-eda)  
   1. [Distributions univariées](#41-distributions-univariées)  
   2. [Répartition de la variable cible](#42-répartition-de-la-variable-cible)  
   3. [Relations bivariées](#43-relations-bivariées)  
   4. [Corrélations entre variables numériques](#44-corrélations-entre-variables-numériques)  
5. [Pré-traitement technique pour le Machine Learning](#5-pré-traitement-technique-pour-le-machine-learning)  
   1. [Séparation features / cible](#51-séparation-features--cible)  
   2. [Pipelines de transformation](#52-pipelines-de-transformation)  
6. [Modélisation et évaluation (structure recommandée)](#6-modélisation-et-évaluation-structure-recommandée)  
7. [Conclusion](#7-conclusion)  
8. [Références (à compléter si nécessaire)](#8-références-à-compléter-si-nécessaire)  

---

## 1. Introduction

L’obésité constitue aujourd’hui un enjeu majeur de santé publique. Elle est associée à une hausse du risque de maladies cardiovasculaires, de diabète de type 2 et de plusieurs complications métaboliques. La disponibilité de données détaillées sur les habitudes alimentaires, l’activité physique et le mode de vie permet de mobiliser les outils de **Data Science** et de **Machine Learning** pour :

- mieux comprendre les facteurs associés aux différents niveaux de corpulence ;
- développer des modèles capables d’**estimer un niveau d’obésité** à partir de caractéristiques individuelles.

Dans le cadre du module **Analyse de données et Machine Learning** (ENCG Settat), ce compte rendu présente une implémentation technique complète en Python, basée sur le dataset UCI :

> **Estimation of Obesity Levels Based on Eating Habits and Physical Condition** *(id=544)*

Les objectifs de travail sont :

1. **Analyser** la structure et le contenu du dataset via un pré-traitement rigoureux et une EDA structurée.
2. **Construire et évaluer** plusieurs modèles de classification supervisée afin de prédire le niveau d’obésité à partir de variables socio‑démographiques, alimentaires et comportementales.

---

## 2. Présentation du dataset

### 2.1 Source et contexte

Le dataset est issu de **l’UCI Machine Learning Repository** et peut être importé en Python via la librairie `ucimlrepo`. L’objectif est de **classer le niveau d’obésité** en fonction d’un ensemble de variables décrivant le mode de vie et les habitudes alimentaires.

Il s’agit donc d’un problème de **classification supervisée multiclasse** : la variable cible comporte plusieurs catégories allant de l’insuffisance pondérale jusqu’à différents niveaux d’obésité.

### 2.2 Unité statistique et taille de l’échantillon

- **Unité statistique :** une observation correspond à **un individu**.
- Les variables décrivent notamment :
  - des informations socio‑démographiques (âge, sexe, taille, etc.) ;
  - des habitudes alimentaires (légumes, aliments caloriques, grignotage, hydratation, alcool, etc.) ;
  - des indicateurs d’activité physique et de mode de vie (sport, temps d’écran, transport, tabagisme, etc.) ;
  - une **classe de niveau d’obésité** (cible).

> **Remarque :** la taille exacte de l’échantillon est obtenue directement dans le notebook via `df.shape` (à conserver comme “source de vérité” des dimensions).

### 2.3 Variable cible

La variable cible correspond au **niveau d’obésité** (souvent notée `NObeyesdad` selon le format du fichier UCI). Les classes observées sont généralement :

- `Insufficient_Weight`
- `Normal_Weight`
- `Overweight_Level_I`
- `Overweight_Level_II`
- `Obesity_Type_I`
- `Obesity_Type_II`
- `Obesity_Type_III`

Ce caractère **multiclasse** (7 catégories) rend l’évaluation plus exigeante qu’une classification binaire, et justifie l’usage de métriques adaptées (F1 macro, matrice de confusion, etc.).

### 2.4 Familles de variables explicatives

Afin de structurer l’analyse, les variables explicatives peuvent être regroupées en familles :

1. **Socio‑démographie**
   - `Age` : âge (numérique)
   - `Height`, `Weight` : taille et poids (numériques)
   - `Gender` : sexe (catégorielle)

2. **Antécédents / perception**
   - `family_history_with_overweight` : historique familial de surpoids
   - `SCC` : suivi/contrôle des calories consommées

3. **Habitudes alimentaires**
   - `FAVC` : consommation fréquente d’aliments très caloriques
   - `FCVC` : fréquence de consommation de légumes
   - `NCP` : nombre de repas principaux
   - `CAEC` : consommation entre les repas (snacking)
   - `CALC` : consommation d’alcool
   - `CH2O` : quantité d’eau bue par jour

4. **Activité physique / mode de vie**
   - `FAF` : fréquence d’activité physique
   - `TUE` : temps d’utilisation des écrans
   - `MTRANS` : mode de transport principal
   - `SMOKE` : statut tabagique

> Une large partie des variables est **catégorielle**, parfois ordinale. Dans l’implémentation, elles sont traitées comme des catégories et encodées par *One-Hot Encoding*.

---

## 3. Méthodologie générale

### 3.1 Environnement technique

L’implémentation est réalisée sous **Google Colab** (Python), en s’appuyant principalement sur :

- `pandas`, `numpy` pour manipulation et préparation des données ;
- `matplotlib`, `seaborn` pour la visualisation ;
- `scikit-learn` pour les pipelines de pré‑traitement, la modélisation et l’évaluation :
  - `train_test_split`, `StratifiedKFold`, `cross_val_score`, `RandomizedSearchCV`
  - `OneHotEncoder`, `StandardScaler`, `ColumnTransformer`, `Pipeline`, `SimpleImputer`
  - `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`
  - `classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay`, `roc_auc_score`

L’import du dataset via `fetch_ucirepo(id=544)` garantit la **reproductibilité** de l’étude.

### 3.2 Pré-traitement des données

Les étapes de pré‑traitement suivies sont :

1. **Chargement et fusion**
   - import des features (`X`) et de la cible (`y`) ;
   - fusion dans un DataFrame `df` pour faciliter l’EDA.

2. **Contrôle des doublons**
   - calcul des duplications (`duplicated().sum()`) ;
   - suppression éventuelle pour limiter les biais d’apprentissage.

3. **Analyse des valeurs manquantes**
   - comptage des valeurs manquantes par variable (`isna().sum()`).
   - mise en place d’une imputation systématique dans les pipelines (voir section 5).

4. **Typage des variables**
   - séparation **numériques** / **catégorielles**, utilisée ensuite dans un `ColumnTransformer`.

### 3.3 Feature engineering

Une variable synthétique est introduite afin de résumer l’information numérique :

- `numeric_risk_score` : somme d’indicateurs binaires indiquant, pour chaque variable numérique, si la valeur d’un individu est **au-dessus de la médiane** observée dans l’échantillon.

**Intérêt :**
- produire un indicateur global simple et interprétable ;
- tester si un score agrégé améliore la capacité prédictive (sans complexifier excessivement le modèle).

### 3.4 Partitionnement des données

Pour évaluer correctement les performances, le dataset est divisé en :

- **Train set :** 80 %
- **Test set :** 20 %

Le partitionnement repose sur :
- `random_state=42` pour reproductibilité ;
- `stratify=y` afin de conserver la distribution des classes entre train et test.

---

## 4. Analyse exploratoire des données (EDA)

### 4.1 Distributions univariées

Des histogrammes (et densités) sont produits pour les variables numériques (ex. âge, poids, taille).  
Ils permettent de repérer :

- des asymétries (skewness) ;
- des valeurs atypiques ;
- des distributions multimodales.

### 4.2 Répartition de la variable cible

Un diagramme en barres de la variable cible met en évidence un **déséquilibre de classes** : certaines catégories (ex. `Normal_Weight`) peuvent être plus fréquentes que d’autres (ex. `Insufficient_Weight`, `Obesity_Type_III`).  
Cette observation encourage l’utilisation de métriques robustes au déséquilibre (F1 macro, rappel/precision par classe).

### 4.3 Relations bivariées

Des boxplots comparent certaines variables numériques selon la classe d’obésité :

- **Poids / IMC vs classe** : augmentation des médianes et quantiles de `Insufficient_Weight` vers `Obesity_Type_III`.
- **Âge vs classe** : variable moins discriminante, mais pouvant montrer des tendances selon les catégories.

Ces visualisations permettent d’identifier les variables les plus informatives.

### 4.4 Corrélations entre variables numériques

Une matrice de corrélation (heatmap) est utilisée pour :

- détecter les corrélations fortes (ex. poids, taille, dérivés d’IMC) ;
- repérer de la multicolinéarité potentielle ;
- guider l’interprétation, notamment pour les modèles linéaires.

---

## 5. Pré-traitement technique pour le Machine Learning

### 5.1 Séparation features / cible

À partir du DataFrame enrichi (`data_fe`) :

- `X_all` regroupe l’ensemble des variables explicatives (incluant `numeric_risk_score`) ;
- `y_all` contient la variable cible multiclasse, typée en `category`.

### 5.2 Pipelines de transformation

Un pré‑traitement différencié est mis en place via `Pipeline` + `ColumnTransformer` :

**A. Pipeline numérique**
- `SimpleImputer(strategy="median")` : imputation par la médiane (robuste aux valeurs extrêmes) ;
- `StandardScaler()` : standardisation, utile pour les modèles sensibles à l’échelle (ex. régression logistique).

**B. Pipeline catégoriel**
- `SimpleImputer(strategy="most_frequent")` : imputation par la modalité la plus fréquente ;
- `OneHotEncoder(handle_unknown="ignore")` : encodage *one‑hot* en évitant les erreurs lors de modalités inconnues en test.

**C. `ColumnTransformer` global**
- fusionne les transformations numériques et catégorielles au sein d’un objet unique `preprocessor`.

---

## 6. Modélisation et évaluation (structure recommandée)

> Cette section est volontairement structurée pour être complétée avec les sorties du notebook (métriques, matrices de confusion, meilleurs hyperparamètres).  
> Elle s’appuie sur la liste d’algorithmes déjà citée dans l’environnement technique.

### 6.1 Modèles testés

- **Régression logistique** (baseline interprétable)  
- **Random Forest** (modèle non linéaire, robuste)  
- **Gradient Boosting** (amélioration progressive par boosting)

### 6.2 Schéma d’évaluation

- **Validation croisée stratifiée** (ex. `StratifiedKFold`) afin de :
  - stabiliser l’estimation des performances ;
  - tenir compte du déséquilibre des classes.
- **Jeu de test** conservé pour l’évaluation finale.

### 6.3 Métriques recommandées (multiclasse)

- Accuracy (à interpréter avec prudence si déséquilibre)
- F1‑score **macro** (plus équitable entre classes)
- Matrice de confusion + rapport de classification (`classification_report`)

### 6.4 Optimisation des hyperparamètres

- `RandomizedSearchCV` pour rechercher efficacement un bon compromis performance/temps de calcul.
- Conservation des meilleurs paramètres et re‑test final sur le jeu de test.

---

## 7. Conclusion

Ce compte rendu présente une démarche complète et reproductible pour analyser un dataset de santé publique, depuis le pré‑traitement jusqu’à la préparation des pipelines destinés au Machine Learning.  
L’étude met l’accent sur :

- la structuration des variables (numériques vs catégorielles) ;
- un pré‑traitement robuste (imputation + normalisation + encodage) ;
- une EDA permettant de comprendre la cible et les relations entre variables ;
- une architecture de modélisation comparable (pipelines + CV + optimisation).

---

## 8. Références (à compléter si nécessaire)

- UCI Machine Learning Repository – Dataset *Estimation of Obesity Levels Based on Eating Habits and Physical Condition* (id=544).  
- Documentation officielle `scikit-learn` : pipelines, encodage, métriques et validation croisée.  
- Documentation `ucimlrepo` : import des datasets UCI via Python.
