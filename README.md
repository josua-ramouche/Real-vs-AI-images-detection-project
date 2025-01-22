# Lancer les Microservices et l'Application en Local

Voilà un petit guide pour lancer tous les microservices en local ! 

## Étapes de démarrage

### 1. Démarrer le serveur Eureka
Lancer l'application **DiscoveryMSApplication** à partir du fichier suivant :  
`discoveryMS/src/main/java/fr/insa/project/DiscoveryMS/DiscoveryMSApplication.java`.

Une fois démarré, assurez-vous que le serveur Eureka est prêt avant de passer à l'étape suivante.

---

### 2. Démarrer l'API Gateway
Lancer l'application **ApiGatewayApplication** à partir du fichier suivant :  
`api-gateway/src/main/java/fr/insa/project/api_gateway/ApiGatewayApplication.java`.

Attendez que **API-GATEWAY** soit correctement enregistré dans le serveur Eureka.

---

### 3. Démarrer le service de prédiction
Lancer l'application **PredictionMSApplication** à partir du fichier suivant :  
`predictionMS/src/main/java/fr/insa/project/PredictionMS/PredictionMSApplication.java`.

Attendez que **PREDICTION-SERVICE** soit correctement enregistré dans le serveur Eureka.

---

### Exécution des requêtes 
Les microservices sont prêts à accueillir des requêtes ! Il faut maintenant lancer le serveur API du code Python (cf. [Lancer le serveur API de prédiction](https://github.com/AurelienPasquet/PI_model_training/tree/predictionAPI?tab=readme-ov-file#lancer-le-serveur-api-de-pr%C3%A9diction))

Pour faire des requêtes, au choix : 
- Lancer l'interface React (cf. [Lancer l'interface React](https://github.com/josua-ramouche/Real-vs-AI-images-detection/tree/local/app#bienvenue-sur-linterface-de-real-vs-fake-sun_with_face))
- Faire une requête depuis Postman : 
![image](https://github.com/user-attachments/assets/c374e744-3304-4663-9727-0afe5a64d43b)

