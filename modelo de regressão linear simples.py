import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dados de exemplo
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Variável independente
y = np.array([2, 4, 5, 4, 5])                # Variável dependente

# Criar um modelo de regressão linear
modelo = LinearRegression()

# Treinar o modelo
modelo.fit(X, y)

# Fazer previsões
previsoes = modelo.predict(X)

# Coeficientes da regressão
coef_angular = modelo.coef_[0]
intercepto = modelo.intercept_

# Plotar os dados e a linha de regressão
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, previsoes, color='red', linewidth=2, label='Regressão Linear')

# Adicionar rótulos e legenda ao gráfico
plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')
plt.title('Regressão Linear Simples')
plt.legend()

# Exibir o gráfico
plt.show()

# Exibir coeficientes
print(f"Coeficiente Angular: {coef_angular}")
print(f"Intercepto: {intercepto}")
