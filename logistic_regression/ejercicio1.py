# Completa el código siguiendo las pistas

# PASO 1: Importa las herramientas necesarias.
# Pista: LogisticRegression, train_test_split,
#        accuracy_score y confusion_matrix.


# PASO 2: Declara X e y utilizando la tabla.
# Pista: X debe ser una matriz (lista de listas).
#        y debe ser una lista de ceros y unos.

# PASO 3: Divide los datos en entrenamiento y prueba.
# Pista: utiliza train_test_split().
#        Reserva el 25 % para prueba.
#        random_state=42 y stratify=y.
# No

# PASO 4: Crea el modelo de regresión logística.
# Pista: utiliza LogisticRegression(max_iter=1000).


# PASO 5: Entrena el modelo.
# Pista: utiliza fit() con las variables
#        del conjunto de entrenamiento.

# PASO 6: Consulta los parámetros aprendidos.
# Pista: muestra coef_ e intercept_.


# PASO 7: Consulta las iteraciones utilizadas.
# Pista: atributo n_iter_.


# PASO 8: Predice la probabilidad de compra
#         de un cliente que estuvo 7.5 minutos.
# Pista: utiliza predict_proba().
#        Obtén la probabilidad de la clase 1.

# Cómo grafico?
# PASO 9: Clasifica al mismo cliente.
# Pista: utiliza predict().


# PASO 10: Realiza predicciones con X_test.
# Pista: guarda el resultado en y_pred.


# PASO 11: Construye la matriz de confusión.
# Pista: compara y_test con y_pred.


# PASO 12: Calcula la exactitud del modelo.
# Pista: utiliza accuracy_score().
