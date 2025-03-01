# Entraîner un modèle sur les GPUs du CSN

## Créer nouvel un environnement conda (depuis votre session INSA)

### Créer l'environnement
```bash
conda create -n torch_env
```

### Activer l'environnement
```bash
conda activate torch_env
```

### Cloner le repo

```bash
git clone https://github.com/AurelienPasquet/PI_model_training.git
```

### Se déplacer dans le repo
```bash
cd PI_model_training/
```

### Installer les packages python
```bash
pip install -r requirements.txt
```

Vous pouvez ensuite continuer sur les machines de l'INSA ou sur votre machine perso.

## Se connecter en SSH sur les machines du CSN

Vous pouvez faire un SSH depuis votre machine perso avec le VPN INSA (option `-X` pour pouvoir voir les plots).

### Serveur 1 :

- Adresse : `srv-gei-gpu1.insa-toulouse.fr`
- Configuration : 2 cartes NVIDIA A45000 avec 21 Go de mémoire chacune.
- **Connection :**
```bash
ssh -X <login_insa>@srv-gei-gpu1.insa-toulouse.fr
```

### Serveur 2 :

- Adresse : `srv-gei-gpu2.insa-toulouse.fr`
- Configuration : 4 cartes NVIDIA P4000 avec 8 Go de mémoire chacune.
- **Connection :**
```bash
ssh -X <login_insa>@srv-gei-gpu2.insa-toulouse.fr
```

### Se déplacer dans le repo
```bash
cd path/to/PI_model_training
```

## Utilisation

### Entraîner un modèle

Créez le dossier `data` et placez votre dataset dedans :

```bash
mkdir data
```

Créez le fichier `config.json`et modifiez les valeurs comme bon vous semble.

Exemple de fichier `config.json` (lancez la commande dans le terminal pour créer le fichier) :

```bash
echo '{
    "epochs": 10,
    "lr": 0.0007,
    "batch_size": 32,
    "input_shape": [3, 32, 32],
    "random_seed": null,
    "model_name": "model_example",
    "model": {
        "conv_block1": [
            {"type": "conv", "params": {"in_channels": 3, "out_channels": 10, "kernel_size": 3, "stride": 1, "padding": 0}},
            {"type": "relu", "params": null},
            {"type": "conv", "params": {"in_channels": 10, "out_channels": 10, "kernel_size": 3, "stride": 1, "padding": 0}},
            {"type": "relu", "params": null},
            {"type": "maxpool", "params": {"kernel_size": 2}}
        ],
        "conv_block2": [
            {"type": "conv", "params": {"in_channels": 10, "out_channels": 10, "kernel_size": 3, "stride": 1, "padding": 0}},
            {"type": "relu", "params": null},
            {"type": "conv", "params": {"in_channels": 10, "out_channels": 10, "kernel_size": 3, "stride": 1, "padding": 0}},
            {"type": "relu", "params": null},
            {"type": "maxpool", "params": {"kernel_size": 2}}
        ],
        "classifier": [
            {"type": "dropout", "params": {"p": 0.5}},
            {"type": "linear", "params": {"out_features": 1}}
        ]
    }
}' > config.json

```

Lancer le programme avec la commande :

```bash
python train.py data/<dataset> config.json
```

### Résultats

Le programme affiche les plots de loss et accuracy ainsi que la matrice de confusion, à vous de les enregistrer/faire une capture d'écran si vous voulez les garder.

Vous trouverez dans le dossier `out/` les fichiers :
- `<model>.pt` : le fichier contenant le modèle.
- `<model>_metrics.json` : le fichier contenant les metrics calculées pendant l'exécution du programme.

### Faire une prédiction sur une image

Créer le dossier `images`:

```bash
mkdir images
```

Copier les images à prédire dans le dossier `images`.
Lancez ensuite une prédiction avec la commande suivante :

```bash
python prediction.py out/<model_name>.pt images/<image>
```

Le programme affiche la classe prédite avec le pourcentage de confiance.

## Lancer le serveur API de prédiction

Depuis un terminal, exécuter la commande suivante à la racine du projet :

```bash
uvicorn api_server:app --reload --host <adresse_ip_serveur> --port <numero_de_port>
```

Pour lancer le serveur en local par exemple : 

```bash
uvicorn api_server:app --reload --host localhost --port 8090
```

Le serveur API tourne et est prêt à accueillir des requêtes HTTP contenant des images !


## Plus d'infos

Si vous voulez en savoir plus sur l'utilité des différents hyperparamètres et sur les CNNs en général, vous pouvez aller voir [ici](https://poloclub.github.io/cnn-explainer/).
