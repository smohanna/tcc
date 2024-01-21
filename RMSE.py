import numpy as np
from sklearn.metrics import mean_squared_error

def calcular_rmse(y_true, y_pred):
    """
    Calcula a Raiz do Erro Quadrático Médio (RMSE).

    Parameters:
    - y_true: Array com os valores reais.
    - y_pred: Array com os valores preditos pelo modelo.

    Returns:
    - rmse: Valor da Raiz do Erro Quadrático Médio.
    """
    # Verifica se os arrays têm o mesmo comprimento
    if len(y_true) != len(y_pred):
        raise ValueError("Os arrays y_true e y_pred devem ter o mesmo comprimento.")

    # Calcula o erro quadrático médio
    mse = mean_squared_error(y_true, y_pred)

    # Calcula a raiz quadrada do erro quadrático médio
    rmse = np.sqrt(mse)

    return rmse

# Exemplo de uso
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.2, 2.5, 3.1, 4.0, 5.3])

rmse_resultado = calcular_rmse(y_true, y_pred)
print(f'A Raiz do Erro Quadrático Médio (RMSE) é: {rmse_resultado}')
