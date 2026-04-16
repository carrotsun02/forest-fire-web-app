from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

app = Flask(__name__)

# 저장된 파이프라인과 모델 불러오기
pipeline = joblib.load('full_pipeline.pkl')
model = keras.models.load_model('forest_fire_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # 사용자가 입력한 데이터 받아오기
        input_data = pd.DataFrame({
            'longitude': [float(request.form['longitude'])],
            'latitude': [float(request.form['latitude'])],
            'month': [request.form['month']],
            'day': [request.form['day']],
            'avg_temp': [float(request.form['avg_temp'])],
            'max_temp': [float(request.form['max_temp'])],
            'max_wind_speed': [float(request.form['max_wind_speed'])],
            'avg_wind': [float(request.form['avg_wind'])]
        })

        # 데이터 전처리 (파이프라인 통과)
        prepared_data = pipeline.transform(input_data)
        
        # 산불 피해 면적 예측
        prediction_log = model.predict(prepared_data)
        
        # y = ln(burned_area + 1) 로 변환했었으므로 다시 지수 변환으로 복구
        # 이 값은 CSV의 원본 수치 단위(예: 아르(are))와 동일합니다.
        predicted_area_raw = np.exp(prediction_log[0][0]) - 1
        predicted_area_raw = max(0, predicted_area_raw)
        
        # divby100 이었으므로 다시 100을 곱해 실제 제곱미터(m²)로 변환
        predicted_area_m2 = predicted_area_raw * 100
        
        return render_template('result.html', prediction_m2=round(predicted_area_m2, 2))

if __name__ == '__main__':
    app.run(debug=True)