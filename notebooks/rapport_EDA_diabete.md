
# 🧭 Mini Rapport EDA — Analyse exploratoire des données du diabète

## 🔹 1. Objectif de l’analyse
L’objectif de cette analyse est d’explorer le jeu de données du diabète afin de mieux comprendre la distribution des variables, détecter les valeurs manquantes ou aberrantes, et préparer les données pour l’entraînement d’un modèle prédictif.

---

## 🔹 2. Aperçu du dataset
Le dataset contient **768 observations** et **9 variables** :
- **Pregnancies** : nombre de grossesses  
- **Glucose** : taux de glucose dans le sang  
- **BloodPressure** : pression artérielle  
- **SkinThickness** : épaisseur de la peau (mm)  
- **Insulin** : taux d’insuline  
- **BMI** : indice de masse corporelle  
- **DiabetesPedigreeFunction** : facteur héréditaire du diabète  
- **Age** : âge du patient  
- **Outcome** : 1 si la personne est diabétique, 0 sinon  

---

## 🔹 3. Statistiques descriptives
Après imputation et nettoyage :
```python
print(df_imputed.describe())
```
➡️ On observe :
- Moyenne du glucose : environ **120 mg/dL**
- Moyenne du BMI : **32**
- Âge moyen : **33 ans**
- Forte variabilité de l’insuline (écart-type élevé)

---

## 🔹 4. Données manquantes et imputation
Certaines variables (comme `Insulin`, `SkinThickness`, `BMI`) contenaient des **valeurs nulles ou nulles remplaçant des zéros**.  
Elles ont été corrigées à l’aide du **KNNImputer (k=7)** pour prédire les valeurs manquantes en se basant sur les observations les plus proches.

```python
imputer = KNNImputer(n_neighbors=7)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

✅ Toutes les valeurs manquantes ont été remplacées avec succès.

---

## 🔹 5. Visualisation des distributions
Les histogrammes montrent la répartition des variables avant et après imputation :

```python
df.hist(bins=20, figsize=(20,15), color="red")
plt.suptitle("Distribution avant imputation")

df_imputed.hist(bins=20, figsize=(20,15), color="green")
plt.suptitle("Distribution après imputation")
```

➡️ Les distributions générales restent cohérentes, mais les valeurs extrêmes sont légèrement lissées.

---

## 🔹 6. Relations entre variables
**Exemple : Glucose vs BMI**
```python
sns.scatterplot(data=df_imputed, x="Glucose", y="BMI", alpha=0.6, color="green")
plt.title("Nuage de points : Glucose vs BMI (après imputation)")
```

📊 On observe une **tendance positive faible** : les individus avec un BMI élevé ont souvent un glucose plus haut.

---

## 🔹 7. Conclusions
- Le dataset est globalement propre et équilibré.  
- Les valeurs manquantes ont été traitées efficacement avec **KNNImputer**.  
- Les distributions sont normales pour la plupart des variables.  
- Certaines relations (Glucose–BMI, Âge–DiabetesPedigreeFunction) peuvent influencer le diagnostic du diabète.  

✅ **Prochaine étape :** normalisation / standardisation des données, puis entraînement des modèles de classification (ex. RandomForest, XGBoost, SVR).
