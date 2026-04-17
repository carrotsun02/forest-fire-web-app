from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
# [중요] 텐서플로우 import는 완전히 삭제했습니다!

app = Flask(__name__)

# 저장된 파이프라인과 초경량 가중치 모델 불러오기
pipeline = joblib.load('full_pipeline.pkl')
weights = joblib.load('model_weights.pkl')

# 3개 층의 가중치(W)와 편향(b)을 각각 변수에 분리
W1, b1, W2, b2, W3, b3 = weights

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
        
        # ----------------------------------------------------
        # [핵심] 텐서플로우 없이 Numpy로 인공신경망 예측 완벽 재현!
        # Layer 1 (Dense 30, ReLU)
        z1 = np.dot(prepared_data, W1) + b1
        a1 = np.maximum(0, z1) 
        
        # Layer 2 (Dense 10, ReLU)
        z2 = np.dot(a1, W2) + b2
        a2 = np.maximum(0, z2)
        
        # Layer 3 (Dense 1, Linear)
        prediction_log = np.dot(a2, W3) + b3
        # ----------------------------------------------------
        
        # y = ln(burned_area + 1) 로 변환했었으므로 다시 지수 변환으로 복구
        predicted_area_raw = np.exp(prediction_log[0][0]) - 1
        predicted_area_raw = max(0, predicted_area_raw)
        
        # divby100 이었으므로 다시 100을 곱해 실제 제곱미터(m²)로 변환
        predicted_area_m2 = predicted_area_raw * 100
        
        return render_template('result.html', prediction_m2=round(predicted_area_m2, 2))

if __name__ == '__main__':
    app.run(debug=True)