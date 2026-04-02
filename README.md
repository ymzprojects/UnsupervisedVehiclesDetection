# Détection de Véhicules et Clustering de Caméras (Non Supervisé)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)

## 📌 Présentation du Projet

Ce dépôt présente un pipeline complet de vision par ordinateur pour la **détection de véhicules non supervisée**. Le projet répond au défi de l'analyse de données de trafic mixtes sans aucune étiquette (labels) manuelle.

Il démontre une stratégie IA modulaire en deux étapes :
1.  **Intelligence Structurelle :** Utilisation du Deep Learning pour regrouper automatiquement les images par point de vue (caméra).
2.  **Détection Statistique :** Utilisation de la vision par ordinateur classique pour isoler les objets mobiles des arrière-plans statiques.

---

## 🚀 Fonctionnalités Clés

### 1. Clustering Automatique de Caméras (PyTorch)
Une solution robuste pour organiser des jeux de données non étiquetés en identifiant les similitudes structurelles entre les différents angles de vue.

* **Extraction de Caractéristiques Profondes :** Implémentation d'un **Auto-encodeur Convolutionnel** pour compresser la géométrie de la scène 2D en un vecteur latent plat.
* **Réduction de Dimension :** Projection des données d'image dans un espace latent qui ignore les objets transitoires (voitures) pour se concentrer sur les structures fixes (routes, bâtiments).
* **Sélection Autonome du K :** Intégration de la **Méthode du Coude** avec `KneeLocator` pour déterminer mathématiquement le nombre optimal de sources vidéo.
* **Regroupement Hiérarchique :** Application du clustering agglomératif sur les signatures latentes pour une isolation précise des points de vue.

### 2. Détection de Véhicules (OpenCV)
Une fois les points de vue isolés, le pipeline extrait les éléments mobiles via la modélisation de l'arrière-plan.

* **Normalisation Globale :** Alignement de l'intensité lumineuse sur l'ensemble du dataset pour atténuer les variations météo et d'exposition.
* **Modélisation par Fond Médian :** Génération d'images de référence "vides" en calculant la médiane de chaque pixel par cluster.
* **Seuillage Dynamique :** Implémentation de la **Binarisation d'Otsu** sur les images de différence pour séparer adaptativement le mouvement du bruit.
* **Raffinement Morphologique :** Opérations d'Ouverture/Fermeture avancées pour consolider les contours des véhicules et éliminer les faux positifs.

---

## 📂 Structure du Dépôt

| Fichier | Description |
| :--- | :--- |
| **`vehicle_detection_pipeline.ipynb`** | **Notebook Principal.** Contient la normalisation, l'entraînement de l'Auto-encodeur, le clustering et la boucle de détection finale. |
| **`opencv.py`** | **Script Modulaire.** Regroupe les fonctions de soustraction de fond, le seuillage d'Otsu et la visualisation des clusters. |

---

## 🛠️ Stack Technique & Méthodologie

### Technologies de Cœur
* **Langages :** Python 3.9+
* **Traitement d'Image :** OpenCV (Logique de détection)
* **Deep Learning :** PyTorch (Architecture Auto-encodeur)
* **Machine Learning :** Scikit-Learn (Clustering & Scaling), Kneed (Optimisation)
* **Analyse de Données :** NumPy, Pandas

### Focus Méthodologique
* **Extraction d'Espace Latent :** Réduction d'images 128x128 en "empreintes digitales" de 64 dimensions pour identifier l'emplacement unique de chaque capteur.
* **Inférence par Lots (Batch) :** Utilisation optimisée des DataLoaders PyTorch pour éviter les erreurs de mémoire GPU (OOM).
* **Heuristiques Non Supervisées :** Détection basée sur la variance temporelle et la cohérence géométrique plutôt que sur des poids de classes pré-entraînés.

---
## 🚀 Installation & Configuration

### Prérequis
* Un compte Google (pour l'utilisation de Google Colab).
* Le fichier `dataset.zip` contenant vos images et le fichier d'annotations.

### Configuration du Google Drive
Pour que le pipeline fonctionne correctement, vous devez préparer votre environnement Drive comme suit :

1. **Téléversement :** Importez votre fichier `dataset.zip` à la racine de votre Google Drive ou dans un dossier spécifique (ex: `MyDrive/Projets/`).
2. **Décompression :** Le notebook inclut une cellule d'extraction automatique. Assurez-vous que le chemin vers le fichier `.zip` dans le code correspond à votre emplacement sur le Drive.
3. **Structure attendue :** Une fois décompressé, le dossier doit contenir :
    * Les images au format `.jpg` ou `.png`.
    * Le fichier `_annotations.csv` regroupant les coordonnées des boîtes englobantes.

### Installation locale
Si vous préférez exécuter le projet hors Colab, clonez le dépôt et installez les dépendances :

```bash
git clone [https://github.com/votre-username/UnsupervisedVehicleDetection.git](https://github.com/votre-username/UnsupervisedVehicleDetection.git)
cd UnsupervisedVehicleDetection

pip install numpy pandas matplotlib opencv-python torch scikit-learn kneed tqdm
pip install numpy pandas matplotlib opencv-python torch scikit-learn kneed tqdm
