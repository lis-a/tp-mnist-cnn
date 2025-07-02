# MNIST – Deep Learning & MLOps

**Auteurs** : AU Lisa, HENRY Dorine

**Classe** : M2 TL - DC Paris

## Présentation du projet

Ce projet a été réalisé dans le cadre du cours de Deep Learning et a pour objectifs :

- D’entraîner un modèle de classification d’images manuscrites (dataset MNIST) avec PyTorch, en comparant notamment un MLP et un CNN.
- De mettre le modèle en production via une API FastAPI et une interface utilisateur Streamlit.
- De dockeriser l’ensemble du projet avec Docker et Docker Compose.
- D’automatiser le déploiement de l’image Docker de l’API grâce à **GitHub Actions**, en respectant les critères du **niveau 1 MLOps** selon Google.

---

## 1. Entraînement et comparaison des modèles

Deux modèles ont été entraînés sur le dataset MNIST :

- Un perceptron multicouche (**MLP**)
- Un réseau de neurones convolutionnel (**CNN**)

### 🔧 Objectif
Comparer les performances des deux approches sur un même volume de paramètres pour mettre en évidence l’intérêt du CNN pour le traitement d’images.

### 📊 Résultats

| Critère             | CNN         | MLP         |
|---------------------|-------------|-------------|
| Nombre de paramètres | ~6.42K      | ~6.44K      |
| Test loss           | 0.0994      | 0.4430      |
| Test accuracy       | 96.93%      | 87.22%      |

### 🔎 Analyse
Le CNN obtient de bien meilleures performances car :
- Il exploite la structure spatiale des images (convolutions, pooling…)
- Il détecte mieux les motifs locaux (traits, contours…)

En comparaison, le MLP considère chaque pixel indépendamment, sans capturer les relations spatiales entre eux.

### ✅ Conclusion
Le CNN généralise mieux sur des images comme celles du MNIST. Il converge plus rapidement, avec une meilleure précision, malgré un nombre de paramètres comparable.

---

## 2. Effet de la permutation des pixels

Une expérience a été menée pour étudier la robustesse des modèles face à une perte de structure spatiale. Pour cela, on a appliqué une **permutation fixe** sur tous les pixels des images MNIST. Cela revient à désorganiser l’image tout en conservant exactement les mêmes valeurs de pixels.

### 🎯 Objectif
Comparer les performances des modèles (CNN vs MLP) **avec des images permutées**.

### 📊 Résultats

| Critère               | CNN (permuté) | MLP (permuté) |
|-----------------------|---------------|---------------|
| Nombre de paramètres | 6.422K        | 6.442K        |
| Test loss            | 0.3948        | 0.4516        |
| Test accuracy        | 88.05%        | 86.92%        |

### 🧠 Analyse

- **Le CNN**, qui dépend fortement de la structure spatiale des images, **voit ses performances fortement baisser** (~9 points de moins qu’en non permuté).
- **Le MLP**, lui, traite déjà les pixels comme un vecteur plat, donc la permutation ne l’affecte presque pas.

### 📌 Conclusion

Lorsque l’information spatiale est perturbée (permutation des pixels), **le CNN perd son avantage** sur le MLP. Cela montre l’importance du format d’entrée pour les architectures à base de convolutions.

---

## 3. Sauvegarde du modèle entraîné

Le modèle CNN entraîné sur les données MNIST non permutées a été sauvegardé pour un usage futur, notamment lors de son déploiement via une API ou une interface.

### 📦 Format utilisé
Le modèle a été sauvegardé au format **PyTorch (.pt)**, contenant les poids du réseau.

### 💾 Code utilisé

```python
# Sauvegarde du modèle CNN entraîné
torch.save(model.state_dict(), "model/mnist-0.0.1.pt")
`````

____

## 4. Déploiement local avec FastAPI

Afin de rendre le modèle accessible à travers une API, nous avons utilisé **FastAPI**, un framework léger et rapide pour la création de services web en Python.

### 🚀 Objectif
Permettre à une application externe (frontend ou script) de faire des prédictions en interrogeant une API HTTP.

### 🔧 Architecture

- Le code de l’API est situé dans `src/app/main.py`
- Le modèle est chargé depuis `model/mnist-0.0.1.pt`
- L’API expose une route `POST /api/v1/predict` pour envoyer une image et obtenir une prédiction.

### 🧪 Test de l’API localement

Pour démarrer l’API localement :

````bash
uvicorn src.app.main:app --reload bash
`````

Une fois lancée, l’interface Swagger est accessible à :
👉 http://localhost:8000/docs

### 📤 Format de requête attendu
L’API attend une image PNG encodée au format multipart/form-data.

### 📥 Format de réponse
L’API retourne la prédiction au format JSON :


````json
{
  "prediction": 4
}
`````

### 🔧 Dépendance nécessaire
Pour gérer les formulaires (envoi de fichiers), l’installation de la librairie suivante est requise :

````bash
pip install python-multipart
`````

----

## 5. Interface web avec Streamlit

Une interface utilisateur a été développée avec **Streamlit** pour permettre de tester le modèle de manière interactive.

### 🎯 Objectif

Permettre à l'utilisateur de dessiner un chiffre à la souris et d'obtenir une prédiction instantanée à partir du modèle entraîné.

### 🧱 Structure

- 📄 `src/app/front.py` : code de l’interface Streamlit
- L’interface appelle l’API FastAPI pour effectuer la prédiction

### 🖌️ Fonctionnalité principale

Grâce au composant `streamlit-drawable-canvas`, l’utilisateur peut dessiner un chiffre à la main (dans un canevas 28x28 pixels) puis l’envoyer à l’API.

### ▶️ Lancer l'interface localement

````bash
streamlit run src/app/front.py
`````

Ensuite, rendez-vous sur :
👉 http://localhost:8501

### 🔧 Dépendances nécessaires
Pour exécuter cette interface, il est nécessaire d’installer les bibliothèques suivantes :

````bash
pip install streamlit streamlit-drawable-canvas
`````

----

## 6. Dockerisation et orchestration avec Docker Compose

### 🎯 Objectif

Faciliter le déploiement du projet en encapsulant l’API FastAPI et l’interface Streamlit dans des conteneurs Docker, orchestrés via `docker-compose`.


#### 🏗️ Construire et lancer l’environnement

````bash
docker compose up --build
`````

L’API est accessible à l’adresse : http://localhost:8000/docs
L’interface Streamlit est disponible sur : http://localhost:8501

---

## 7. 🏗️ Industrialisation du projet
Afin de préparer le projet à une mise en production robuste et maintenable, nous avons structuré le dépôt en suivant une architecture claire, inspirée des bonnes pratiques MLOps :

````bash
TP1/
├── model/                      # Modèle entraîné au format .pt
├── data/                      # Données MNIST téléchargées
│   └── raw/                   # Données brutes
├── notebook/                  # Fichiers exploratoires Jupyter
│   └── init.ipynb
├── src/                       # Code source Python
│   ├── model/                 # Script d'entraînement du modèle
│   │   └── train_model.py
│   └── app/                   # Application API + Frontend
│       ├── main.py            # Code de l'API FastAPI
│       ├── front.py           # Interface utilisateur Streamlit
│       ├── Dockerfile.api     # Dockerfile pour l'API
│       └── Dockerfile.front   # Dockerfile pour le front
├── docker-compose.yml         # Orchestration des services API/Front
├── .gitignore                 # Fichiers à exclure du suivi Git
└── README.md                  # Présentation du projet

`````

### ⚙️ Principes appliqués

- **Séparation des rôles** :
  - Le code de data science exploratoire (`notebook/`) est isolé du code de production (`src/`).
  - Le modèle est sauvegardé dans un dossier à part (`model/`) pour faciliter les déploiements.

- **Modularité** :
  - L’API (`main.py`) est indépendante du front (`front.py`) et chacun a son propre Dockerfile.

- **Prêt pour le déploiement** :
  - Grâce à `docker-compose.yml`, le projet est déployable localement ou en production en une seule commande.
  - Les dépendances sont isolées dans des conteneurs.

- **MLOps niveau 1** :
  - Le déploiement du modèle est automatisé via un **workflow GitHub Actions** qui construit et pousse l'image Docker sur Docker Hub après chaque commit.


---

## 8. Déploiement automatique avec GitHub Actions (MLOps Niveau 1)

### 🎯 Objectif

Déployer automatiquement l'image Docker de l’API à chaque mise à jour du code sur GitHub.  
Cela répond au **niveau 1** des pratiques MLOps selon Google :  
> Automatiser le déploiement des modèles en production.

### 🛠️ Étapes réalisées

1. Création d’un dépôt distant sur [Docker Hub](https://hub.docker.com/)
2. Ajout d’un fichier GitHub Actions (`.github/workflows/docker-image.yml`) à la racine du projet
3. Stockage des **secrets Docker Hub** (`DOCKER_USERNAME`, `DOCKER_PASSWORD`) dans les **Secrets GitHub**
4. À chaque `git push`, le workflow :
   - construit l’image Docker à partir du `Dockerfile.api`
   - la pousse sur Docker Hub

Une fois configuré, chaque commit déclenche automatiquement le pipeline de déploiement.

--- 
## 9. Bilan et pistes d’amélioration

### ✅ Résultats obtenus

- **Modèle entraîné avec succès** sur le dataset MNIST
- **Comparaison CNN vs MLP** : meilleure précision pour le CNN (~97%)
- **Déploiement local fonctionnel** via **FastAPI** et interface avec **Streamlit**
- **Automatisation** du déploiement Docker avec **GitHub Actions**

### 📈 Améliorations possibles

- **Augmentation de données (Data Augmentation)** :
  - Rotation, zoom, translation pour rendre le modèle plus robuste
- **Modèle plus complexe** :
  - Ajouter des couches, augmenter le nombre de kernels
- **Évaluation plus fine** :
  - Matrice de confusion, précision par chiffre
- **Meilleure UX dans le frontend** :
  - Prévisualisation claire du dessin
  - Indication de confiance de la prédiction
- **Ajout de tests automatisés** :
  - Pour vérifier les performances du modèle ou le bon fonctionnement de l'API
- **Suivi de version du modèle** :
  - Avec des outils comme DVC ou MLflow (niveau MLOps 2+)

---

🎓 Ce TP constitue une première mise en œuvre des principes MLOps, en partant de l'entraînement jusqu’au déploiement continu d’un modèle simple.  
Il pourra être réutilisé comme base pour des cas plus complexes.
