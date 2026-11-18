from model import predict_with_pipeline

# Exemple d'entrée utilisateur
user_input = [
    1058.0, 3522485.0, 27.0, 84.0, 117.0,
    1364.0, 1.0, 80.0, 0.4106, 0.6722,
    0.8846, 0.0191, 0.6842, 0.8518, 0.0,
    1.415, -0.1154, -0.2415, 0.8778
]

# Appeler la fonction de prédiction
predictions = predict_with_pipeline(user_input)
print(predictions) 