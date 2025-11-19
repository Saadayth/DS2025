# AIT BAHA Saad - Etudiant Groupe 1 CAC

<img src="https://github.com/user-attachments/assets/1fc30205-0df3-4b78-82c5-246950d26dd6" alt="173" width="200"/>


# Analyse du Dataset Wine Quality – Rapport Markdown

## 1. Description du Dataset

Le dataset **Wine Quality – White Wine** provient de l’UCI Machine Learning Repository.  
Il décrit des vins blancs portugais à l’aide de **11 caractéristiques physico-chimiques** :

- fixed acidity  
- volatile acidity  
- citric acid  
- residual sugar  
- chlorides  
- free sulfur dioxide  
- total sulfur dioxide  
- density  
- pH  
- sulphates  
- alcohol  

La variable cible `quality` est une note entre **3 et 9** attribuée par des experts.

Pour l’analyse, la qualité a été convertie en variable **binaire** :
- 0 → qualité ≤ 5 (mauvais ou moyen vin)  
- 1 → qualité > 5 (bon vin)

---

## 2. Analyse Statistique des Variables

### 2.1 Boxplots des caractéristiques physico-chimiques

Ces boxplots montrent l’échelle, la dispersion et les valeurs extrêmes pour chaque variable.  
Certaines variables présentent de nombreux outliers (ex : *total sulfur dioxide*, *free sulfur dioxide*, *residual sugar*).

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/a1c3ed9f-a768-4fbd-8b61-d7abed672a93" />

---

### 2.2 Matrice de Corrélation

La matrice de corrélation révèle des relations notables :
- Corrélation positive entre **sulfur dioxide** (free & total).  
- Corrélation entre **density** et **residual sugar**.  
- Corrélation négative entre **alcohol** et **density**.

<img width="939" height="790" alt="image" src="https://github.com/user-attachments/assets/c9814c87-7960-44ab-a15e-15945c163f59" />

---

## 3. k-NN Sans Normalisation

Le modèle k-NN a été testé pour **k ∈ {1,3,5,…,39}**.

Observations :
- Pour **k=1**, l’erreur d’apprentissage est nulle → **sur-apprentissage**.  
- L’erreur de validation reste élevée (~0.32).  
- Les deux courbes divergent fortement → mauvais généralisation.

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/90eb2e19-0e00-42b2-b226-c034bb9d10f8" />

---

## 4. k-NN Avec Normalisation

Après normalisation (centrage-réduction), les performances s’améliorent :

- L’erreur de validation diminue (~0.24).  
- Les courbes apprentissage/validation sont plus proches.  
- Le modèle généralise mieux.

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/8eed20a1-c38f-4a6d-8d23-4ca07163e375" />

---

## 5. Conclusion Générale

L’étude montre que :

- Le dataset présente des variables sur des **échelles très différentes**, ce qui pénalise les méthodes basées sur les distances.  
- Le modèle k-NN sans normalisation souffre fortement de **sur-apprentissage**.  
- La normalisation permet une nette amélioration des performances et une meilleure stabilité du modèle.
- Le choix optimal de k doit être fait via un ensemble de validation, car trop petit k → variance élevée, trop grand k → biais trop fort.

En résumé, la normalisation est **indispensable** pour appliquer correctement k-NN sur ce dataset, et elle permet d’obtenir une erreur de test nettement plus faible.

