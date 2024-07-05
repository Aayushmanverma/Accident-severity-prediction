from flask import Flask, jsonify, request
import pickle
import numpy as np
import requests
from flask_cors import CORS
from PyPDF2 import PdfReader
import logging
import io

app = Flask(__name__)

CORS(app)

logging.basicConfig(level=logging.DEBUG)
    
@app.route('/api/accident_pred', methods = ['POST'])
def get_details():
    # return jsonify(books)
    data = request.json
    
    # Define default healthy values
    default_values = {
        'did_Police_Officer_Attend_Scene_of_Accident': 50,
        'age_of_Driver': 30,
        'vehicle_Type': 120,
        'age_of_Vehicle': 200,
        'engine_Capacity': 100,
        'day_of_Week': 0,
        'weather_Conditions': 150,
        'road_Surface_Conditions': 0,
        'light_Conditions': 0,
        'sex_of_Driver': 0,
        'speed_limit': 0,
    }
    print("start")
    
    # Convert values to float and use default if missing or empty
    did_Police_Officer_Attend_Scene_of_Accident = float(data.get('did_Police_Officer_Attend_Scene_of_Accident', default_values['did_Police_Officer_Attend_Scene_of_Accident']) or default_values['did_Police_Officer_Attend_Scene_of_Accident'])
    age_of_Driver = float(data.get('age_of_Driver', default_values['age_of_Driver']) or default_values['age_of_Driver'])
    vehicle_Type = float(data.get('vehicle_Type', default_values['vehicle_Type']) or default_values['vehicle_Type'])
    age_of_Vehicle = float(data.get('age_of_Vehicle', default_values['age_of_Vehicle']) or default_values['age_of_Vehicle'])
    engine_Capacity = float(data.get('engine_Capacity', default_values['engine_Capacity']) or default_values['engine_Capacity'])
    day_of_Week= float(data.get('day_of_Week', default_values['day_of_Week']) or default_values['day_of_Week'])
    weather_Conditions = float(data.get('weather_Conditions', default_values['weather_Conditions']) or default_values['weather_Conditions'])
    road_Surface_Conditions = float(data.get('road_Surface_Conditions', default_values['road_Surface_Conditions']) or default_values['road_Surface_Conditions'])
    light_Conditions = float(data.get('light_Conditions', default_values['light_Conditions']) or default_values['light_Conditions'])
    sex_of_Driver = float(data.get('sex_of_Driver', default_values['sex_of_Driver']) or default_values['sex_of_Driver'])
    speed_limit = float(data.get('speed_limit', default_values['speed_limit']) or default_values['speed_limit'])
    # thal = float(data.get('thal', default_values['thal']) or default_values['thal'])
    
    # # Determine the value of 'sex'
    # sex = 1 if data.get('sex', '').lower() == 'male' else 0

    print(data)
    
    with open('accident_model.pkl', 'rb') as f:
        model = pickle.load(f)

    input_data = (did_Police_Officer_Attend_Scene_of_Accident,age_of_Driver,vehicle_Type,age_of_Vehicle,engine_Capacity,day_of_Week,weather_Conditions,road_Surface_Conditions,light_Conditions,sex_of_Driver,speed_limit)

    # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    print(prediction)
    print("end")
    return jsonify({"result" : prediction[0]})
    

def debugger():
    with open('accident_model.pkl', 'rb') as f:
        model = pickle.load(f)

    input_data = (1,10,8300,11,12,3,4,5,8,9,30)

    # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    print(prediction)

if __name__ == '__main__':
    app.run(debug=True)
    # debugger()
