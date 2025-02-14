import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Загрузка данных
data = pd.read_csv('/content/parkinsons.data')

# Просмотр первых нескольких строк
print(data.head())

# Проверка на наличие пропущенных значений
print(data.isnull().sum())

# Отделение признаков от целевой переменной
X = data.drop(['name', 'status'], axis=1)
y = data['status']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#for compatibility
#!pip uninstall -y scikit-learn
#!pip install scikit-learn==1.3.1

# Создание и обучение модели XGBoost
model = XGBClassifier( eval_metric='logloss')
model.fit(X_train_scaled, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test_scaled)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Матрица ошибок и отчет о классификации
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))