# app.py - ПРОСТЕЙШЕЕ ВЕБ-ПРИЛОЖЕНИЕ
# График - ваш готовый скриншот

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Загружаем модель
model = joblib.load('my_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    temperature = float(request.form['temperature'])
    profit = model.predict([[temperature]])[0]

    return render_template('result.html',
                           temperature=temperature,
                           profit=round(profit, 2))


if __name__ == '__main__':
    app.run(debug=True)