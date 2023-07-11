from flask import Flask, request, jsonify
import os
import pickle
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

# 1. Endpoint que devuelva la predicción de los nuevos datos enviados mediante argumentos en la llamada
@app.route('/v1/predict', methods=['GET'])
def predict():
    model = pickle.load(open('data/advertising_model','rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[int(tv),int(radio),int(newspaper)]])
        return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'

# Creamos la base de datos.

df = pd.read_csv("data/Advertising.csv",index_col=0)
df['newpaper'] = df['newpaper'].str.replace("s","")
df['newpaper'] = df['newpaper'].astype(float)
connection = sqlite3.connect("base_datos.db")
df.to_sql('base_datos', connection, if_exists='append',index=False)
connection.close()

# Creamos la función para añadir nuevos registros.
@app.route("/v2/ingest_data", methods=["POST"])
def nuevos_registros():
    connection = sqlite3.connect("base_datos.db")
    cursor = connection.cursor()
    data = {}
    data['TV'] = request.args.get('tv')
    data['radio'] = request.args.get('radio')
    data['newpaper'] = request.args.get('newpaper')
    data['sales'] = request.args.get('sales')
    df = pd.read_sql_query("SELECT * FROM base_datos", connection)
    df = df.append(data, ignore_index=True)
    df.to_sql('base_datos', connection, if_exists='replace', index=False)
    connection.close()

    return jsonify(df.to_dict(orient='records'))

@app.route("/v3/retrain", methods=["POST"])
def retrain_model():
    model = pickle.load(open('data/advertising_model','rb'))
    connection = sqlite3.connect("base_datos.db")
    df = pd.read_sql_query("SELECT * FROM base_datos", connection)
    connection.close()

    # Reentrena el modelo con los datos actualizados
    X = df[['TV', 'radio', 'newpaper']]
    y = df['sales']
    model2 = model.fit(X, y)

    # Guarda el modelo reentrenado
    pickle.dump(model2, open('data/new_model', 'wb'))

    return "Model retrained successfully"


app.run()

