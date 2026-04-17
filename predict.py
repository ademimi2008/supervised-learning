import joblib
import pandas as pd

model = joblib.load('my_model.pkl')

while True:
    temp_input = input("Введите температуру (°F): ")

    try:
        temp = float(temp_input)
        break
    except:
        print("Ошибка! Введите число)")
        print("Попробуйте еще раз:")

input_data = pd.DataFrame([[temp]], columns=['Temperature'])
profit = model.predict(input_data)[0]

print(f"При {temp}°F прибыль: ${profit:.2f}")

if profit < 30:
    print("Совет: берите мало")
elif profit < 50:
    print("Совет: берите среднюю партию")
else:
    print("Совет: берите много")