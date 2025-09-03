import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  

# Supondo que você tenha um conjunto de dados com X como features e y como rótulos/targets
X = np.array
y = np.array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Treine o seu modelo (substitua LogisticRegression pelo seu modelo)
model = LogisticRegression()
model.fit(X_train, y_train)

# Faça previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcule a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

# Calcule o F1 Score
f1 = f1_score(y_test, y_pred)

print("Matriz de Confusão:")
print(conf_matrix)
print(f"F1 Score: {f1}")
