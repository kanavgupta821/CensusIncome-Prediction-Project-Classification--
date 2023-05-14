from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    
    else:
        data=CustomData(
            age=float(request.form.get('age')),
            workclass=request.form.get('workclass'),
            fnlwgt=float(request.form.get('fnlwgt')),
            education=request.form.get('education'),
            marital_status=request.form.get('marital_status'),
            occupation=request.form.get('occupation'),
            relationship=request.form.get('relationship'),
            race=request.form.get('race'),
            sex=request.form.get('sex'),
            capital_gain=float(request.form.get('capital_gain')),
            capital_loss=float(request.form.get('capital_loss')),
            hours_per_week=int(request.form.get('hours_per_week')),
            native_country=request.form.get('native_country')
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0")
