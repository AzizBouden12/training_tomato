# Projet universitaire - Détection et classification automatique de maladies des plantes

## Livrable principal

Le livrable principal demandé par l'enseignant est le notebook Jupyter:

- [01_projet_pas_a_pas.ipynb](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/notebooks/01_projet_pas_a_pas.ipynb)

Ce notebook a été réorganisé pour être exécuté **cellule par cellule** et couvrir proprement les étapes du sujet:

1. vérification du dataset,
2. prétraitement,
3. histogrammes et analyse visuelle,
4. segmentation et contours,
5. extraction de caractéristiques,
6. classification par Machine Learning,
7. comparaison avec le Deep Learning.

## Choix du projet

Le projet travaille sur **4 classes** du dataset PlantVillage, ce qui correspond bien au format individuel:

- `Tomato___healthy`
- `Tomato___Early_blight`
- `Tomato___Late_blight`
- `Tomato___Leaf_Mold`

Pourquoi ce choix:

- les 4 classes sont cohérentes et faciles à défendre pendant la discussion,
- le dataset est déjà préparé en `train/val/test`,
- la comparaison ML / DL est lisible,
- le temps d'exécution reste raisonnable.

## Structure utile du dossier

- [notebooks/01_projet_pas_a_pas.ipynb](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/notebooks/01_projet_pas_a_pas.ipynb): notebook final à rendre
- [prepared_tomato_data](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/prepared_tomato_data): dataset déjà préparé
- [artifacts_final](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/artifacts_final): métriques et figures finales
- [train_tomato_models.py](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/train_tomato_models.py): pipeline Deep Learning propre
- [run_project.ps1](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/run_project.ps1): relance l'entraînement DL validé
- [launch_interface.py](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/launch_interface.py): interface locale de test
- [run_interface.ps1](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/run_interface.ps1): lance l'interface

## Résultats vérifiés

Les résultats ci-dessous ont été vérifiés dans le projet actuel.

### Machine Learning classique

Le notebook construit un vecteur de **65 features** basé sur:

- histogrammes RGB,
- histogrammes HSV,
- texture GLCM,
- descripteurs de forme,
- statistiques simples sur la feuille segmentée.

Deux modèles classiques sont comparés sur le jeu de validation:

- `RandomForest`
- `LinearSVC`

Le meilleur modèle retenu est `RandomForest`.

Résultats test finaux:

- Accuracy: `0.9029`
- Macro F1: `0.8962`
- Weighted F1: `0.9012`

Fichiers associés:

- [artifacts_final/classical_ml/metrics.json](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/artifacts_final/classical_ml/metrics.json)
- [artifacts_final/classical_ml/confusion_matrix.png](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/artifacts_final/classical_ml/confusion_matrix.png)
- [artifacts_final/classical_ml/validation_model_selection.csv](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/artifacts_final/classical_ml/validation_model_selection.csv)

### Deep Learning

Le modèle Deep Learning validé est un `MobileNetV3 Transfer`.

Résultats test vérifiés:

- Accuracy: `0.9570`
- Macro F1: `0.9540`
- Weighted F1: `0.9563`

Fichiers associés:

- [artifacts_final/summary.json](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/artifacts_final/summary.json)
- [artifacts_final/mobilenet_v3_transfer/history.png](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/artifacts_final/mobilenet_v3_transfer/history.png)
- [artifacts_final/mobilenet_v3_transfer/confusion_matrix.png](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/artifacts_final/mobilenet_v3_transfer/confusion_matrix.png)

### Comparaison finale

Le tableau final est enregistré ici:

- [artifacts_final/final_comparison.csv](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/artifacts_final/final_comparison.csv)

Conclusion technique:

- le pipeline classique fonctionne correctement et répond au sujet,
- le Deep Learning donne les meilleures performances finales,
- le gain du DL est d'environ `+5.4%` en accuracy test sur ce projet.

## Comment exécuter le projet

### Option 1 - Rendu principal recommandé

Ouvrir Jupyter dans le dossier du projet puis lancer:

- [notebooks/01_projet_pas_a_pas.ipynb](/C:/Users/gharsalli%20hind/Desktop/Projet_Ing_img/notebooks/01_projet_pas_a_pas.ipynb)

Exécuter les cellules dans l'ordre.

Le notebook:

- recharge le cache de features s'il existe,
- reconstruit les métriques ML,
- peut piloter l'entraînement DL depuis une cellule dédiée,
- recharge ensuite les résultats DL validés,
- produit la comparaison finale.

#### Stratégie Deep Learning dans le notebook

Le notebook contient un paramètre `DL_MODE` avec 3 usages:

- `reuse`: ne relance pas l'entraînement et recharge les artefacts existants
- `fast`: relance un entraînement court pour vérifier tout le pipeline depuis le notebook
- `full`: relance l'entraînement complet pour le rendu final

Cette stratégie évite les blocages inutiles tout en gardant **un seul notebook final** pour piloter tout le projet.

### Option 2 - Relancer le pipeline Deep Learning

```powershell
python -m pip install -r requirements.txt
./run_project.ps1
```

### Option 3 - Tester le modèle via l'interface

```powershell
./run_interface.ps1
```

Ou en ligne de commande:

```powershell
python launch_interface.py --artifact-dir artifacts_final --smoke-test "prepared_tomato_data\test\Tomato___Late_blight\005e3b43-9050-47da-9498-f9ecdcc703b3___RS_Late.B 5104__329190e059.JPG"
```

## Dépendances

Installer les dépendances avec:

```powershell
python -m pip install -r requirements.txt
```

## Évaluation honnête par rapport au cahier des charges

### Ce qui est maintenant bien couvert

- 4 classes conformes au format individuel
- prétraitement expliqué et visualisé
- histogrammes d'images
- segmentation et contours
- extraction de caractéristiques couleur / texture / forme
- modèle Machine Learning classique
- métriques `accuracy`, `precision`, `recall`, `F1`
- comparaison avec un modèle Deep Learning plus avancé
- notebook propre comme livrable principal

### Ce qu'il faut dire clairement pendant la discussion

- la segmentation montrée est **comparative et visuelle**, pas une vérité absolue;
- le pipeline classique reste interprétable mais moins performant que le DL;
- le modèle DL réutilisé dans le notebook est déjà entraîné pour éviter un rendu trop long, mais son pipeline d'entraînement reste fourni dans le projet.

### Est-ce que le projet peut avoir la note complète ?

Honnêtement: **personne ne peut garantir 20/20** sans voir la grille exacte de l'enseignant et la qualité de la discussion finale.

Mais techniquement, le projet est maintenant **beaucoup mieux aligné** avec le sujet qu'avant. Avec un notebook propre, une explication claire des choix, et une discussion correcte des résultats, il peut viser une **très bonne note**.

Le principal risque restant n'est plus le code, mais:

- la qualité de l'explication,
- la justification des choix,
- la manière de défendre la comparaison ML vs DL.

## Nettoyage effectué

Le projet a été recentré autour:

- d'un seul notebook principal,
- d'un seul dossier d'artefacts finaux,
- d'une documentation plus claire,
- de résultats ML et DL réellement vérifiés.
