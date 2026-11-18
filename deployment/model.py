import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Chargement des modèles et du pipeline
scaler = joblib.load('model_data/scaler.pkl')
pca = joblib.load('model_data/pca.pkl')
models = {
    "KNN": joblib.load('model/KNN_model.pkl'),
    "SVM": joblib.load('model/SVM_model.pkl'),
    "DecisionTree": joblib.load('model/DecisionTree_model.pkl'),
    "RandomForest": joblib.load('model/RandomForest_model.pkl'),
    "XGBoost": joblib.load('model/XGBoost_model.pkl'),
    "LogisticRegression": joblib.load('model/LogisticRegression_model.pkl'),
    "NaiveBayes": joblib.load('model/NaiveBayes_model.pkl'),
    "ANN": 'model/model.weights.h5'  # Chemin du modèle ANN
}

def load_ann_model(weights_path):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Input
    
    model = Sequential()
    
    model.add(Input(shape=(12,)))  
    model.add(Dense(128, activation='relu'))  
    model.add(Dropout(0.2))  
    
    
    model.add(Dense(256, activation='relu'))  
    model.add(Dropout(0.3))  
    
    
    model.add(Dense(7, activation='sigmoid'))  
    
    model.load_weights(weights_path)
    
    return model



ann_model = load_ann_model(models["ANN"])

# Pipeline de prétraitement
pipeline = Pipeline([
    ('scaler', scaler),
    ('pca', pca)
])

target_mapping = {
    0: "Bumps",
    1: "Dirtiness",
    2: "K_Scatch",
    3: "Other_Faults",
    4: "Pastry",
    5: "Stains",
    6: "Z_Scratch"
}

def map_prediction(prediction_encoded):
    # Remplacer l'index par le nom du target
    return target_mapping.get(prediction_encoded, "Classe inconnue")


def predict_with_pipeline(user_input):
    # Convertir user_input en un tableau 2D avec des noms de caractéristiques
    user_input_df = pd.DataFrame([user_input], columns=[
        'X_Minimum', 'Y_Minimum', 'Y_Perimeter', 'Minimum_of_Luminosity',
        'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
        'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
        'Square_Index', 'Outside_X_Index', 'Edges_X_Index',
        'Edges_Y_Index', 'Outside_Global_Index', 'Log_X_Index',
        'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'
    ])
    
    transformed_data = pipeline.transform(user_input_df)  

    predictions = {}
    for model_name, model in models.items():
        if model_name == "ANN":
            prediction_proba = ann_model.predict(transformed_data)
            prediction_encoded = np.argmax(prediction_proba)  
        else:
            prediction_encoded = model.predict(transformed_data)[0]
        predictions[model_name] = map_prediction(prediction_encoded)

    return predictions


def get_model_names():
    return list(models.keys())
