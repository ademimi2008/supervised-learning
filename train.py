import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

data = pd.read_csv('Ice Cream Sales - temperatures.csv')

X = data[['Temperature']]
y = data['Ice Cream Profits']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred) * 100

print(f"Точность: {accuracy:.2f}%")

joblib.dump(model, 'my_model.pkl')
print("Модель сохранена")

plt.figure(figsize=(10, 6))

plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Реальные данные')

X_sorted = X_test.sort_values(by='Temperature')
y_sorted = model.predict(X_sorted)
plt.plot(X_sorted, y_sorted, color='red', linewidth=2, label='Линия регрессии')

plt.xlabel('Температура (°F)')
plt.ylabel('Прибыль ($)')
plt.title(f'Линейная регрессия (Точность: {accuracy:.1f}%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('my_model.png')
plt.show()