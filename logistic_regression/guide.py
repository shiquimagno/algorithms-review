# PASO 1: Importar las bibliotecas necesarias

# Algoritmo de regresión logística para clasificación.
from sklearn.linear_model import LogisticRegression

# Métricas para evaluar las predicciones del modelo.
from sklearn.metrics import accuracy_score, confusion_matrix

# Función para dividir datos de entrenamiento y prueba.
from sklearn.model_selection import train_test_split

# PASO 2: Definir los datos y las etiquetas reales

# X: características de entrada (horas de estudio).
# Cada lista interna representa un estudiante.
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

# y: etiquetas reales conocidas.
# 0 = desaprobó, 1 = aprobó.
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


# PASO 3: Dividir los datos en entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,  # 30 % para prueba y 70 % para entrenamiento.
    random_state=42,  # Permite reproducir la misma división.
    stratify=y,  # Mantiene la proporción de clases 0 y 1.
)


# PASO 4: Crear el modelo de regresión logística

# max_iter: número máximo de iteraciones del optimizador.
# Todavía no se realiza ningún entrenamiento.
modelo = LogisticRegression(max_iter=1000)


# PASO 5: Entrenar el modelo

# fit() aprende los pesos y el sesgo usando X_train e y_train.
# Internamente:
# 1. Inicializa los parámetros.
# 2. Calcula z = w*x + b (combinación lineal).
# 3. Aplica la sigmoide para obtener probabilidades.
# 4. Calcula una función de pérdida con las etiquetas reales.
# 5. Optimiza los parámetros y repite hasta converger.
modelo.fit(X_train, y_train)


# PASO 6: Consultar los parámetros aprendidos

# coef_: pesos asignados a las características.
# intercept_: sesgo de la combinación lineal.
# Ambos son parámetros aprendidos durante fit().
print("Pesos:", modelo.coef_)
print("Sesgo:", modelo.intercept_)


# PASO 7: Consultar las iteraciones realizadas

# n_iter_: iteraciones realizadas por el optimizador.
# No equivale necesariamente al número de épocas.
print("Iteraciones:", modelo.n_iter_)


# PASO 8: Calcular la probabilidad de aprobar

# predict_proba() calcula las probabilidades de cada clase.
# [[5.5]]: estudiante que estudió 5.5 horas.
# [0][1]: primera observación, probabilidad de clase 1.
probabilidad = modelo.predict_proba([[5.5]])[0][1]

print("Probabilidad:", probabilidad)


# PASO 9: Clasificar un estudiante nuevo

# predict() aplica la regla de clasificación del modelo.
# Para clasificación binaria estándar, usa umbral de 0.5.
# Devuelve 0 (desaprueba) o 1 (aprueba).
prediccion = modelo.predict([[5.5]])

print("Predicción:", prediccion[0])


# PASO 10: Predecir los datos de prueba

# X_test contiene estudiantes no utilizados en entrenamiento.
# Sus clases predichas se guardan en y_pred.
y_pred = modelo.predict(X_test)


# PASO 11: Obtener la matriz de confusión

# Compara etiquetas reales (y_test) con predicciones (y_pred).
# Orden predeterminado para estas clases: [[VN, FP], [FN, VP]].
# VN: verdadero negativo; FP: falso positivo.
# FN: falso negativo; VP: verdadero positivo.
matriz = confusion_matrix(y_test, y_pred)

print("Matriz de confusión:\n", matriz)


# PASO 12: Calcular la exactitud del modelo

# accuracy = predicciones correctas / total de predicciones.
# Por ejemplo, 0.80 significa 80 % de aciertos.
exactitud = accuracy_score(y_test, y_pred)

print("Exactitud:", exactitud)
