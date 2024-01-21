import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Gerar pontos aleatórios como localizações de coleta
np.random.seed(42)
num_locations = 10
locations = np.random.rand(num_locations, 2) * 10

# Calcular a matriz de distância entre os pontos
dist_matrix = distance_matrix(locations, locations)

# Resolver o Problema do Caixeiro Viajante usando a Atribuição Linear (Linear Sum Assignment)
row_ind, col_ind = linear_sum_assignment(dist_matrix)

# Reorganizar os pontos de acordo com a solução otimizada
optimized_route = locations[col_ind]

# Adicionar o ponto inicial ao final para fechar a rota
optimized_route = np.vstack((optimized_route, optimized_route[0]))

# Visualizar a rota otimizada
plt.plot(optimized_route[:, 0], optimized_route[:, 1], marker='o', linestyle='-')
plt.scatter(locations[:, 0], locations[:, 1], color='red', marker='x')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Rota Otimizada para Coleta')
plt.show()

