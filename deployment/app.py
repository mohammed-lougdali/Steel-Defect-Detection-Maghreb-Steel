from flask import Flask, render_template, request, jsonify
from model import predict_with_pipeline, get_model_names
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    model_names = get_model_names()
    return render_template('index.html', model_names=model_names)

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        # try : 
            user_input = [
            float(request.form.get('X_Minimum', 0)),
            float(request.form.get('Y_Minimum', 0)),
            float(request.form.get('Y_Perimeter', 0)),
            float(request.form.get('Minimum_of_Luminosity', 0)),
            float(request.form.get('Maximum_of_Luminosity', 0)),
            float(request.form.get('Length_of_Conveyer', 0)),
            float(request.form.get('TypeOfSteel_A300', 0)),
            float(request.form.get('Steel_Plate_Thickness', 0)),
            float(request.form.get('Edges_Index', 0)),
            float(request.form.get('Empty_Index', 0)),
            float(request.form.get('Square_Index', 0)),
            float(request.form.get('Outside_X_Index', 0)),
            float(request.form.get('Edges_X_Index', 0)),
            float(request.form.get('Edges_Y_Index', 0)),
            float(request.form.get('Outside_Global_Index', 0)),
            float(request.form.get('Log_X_Index', 0)),
            float(request.form.get('Orientation_Index', 0)),
            float(request.form.get('Luminosity_Index', 0)),
            float(request.form.get('SigmoidOfAreas', 0)),
        ]
            predictions = predict_with_pipeline(user_input)
        
            return render_template('index.html', predictions=predictions)
        #except:
         #   return render_template('index.html')
        # Convertir user_input en un tableau 2D

        # Renvoyer les résultats à la page HTML
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
