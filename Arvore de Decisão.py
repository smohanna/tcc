# Importando as bibliotecas necessárias
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Exemplo: Classificação binária (0 ou 1)
data = pd.read_csv('mismagenedwaste.csv')
features = [[0, 0], [1, 1], [1, 0], [0, 1]]
labels = [0, 1, 1, 0]

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# Criando o classificador de árvore de decisão
clf = tree.DecisionTreeClassifier()

# Treinando o classificador
clf.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
predictions = clf.predict(X_test)

# Calculando a acurácia do modelo
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia: {accuracy}')

# Visualizando a árvore de decisão (opcional)
# (Este passo pode exigir a instalação da biblioteca graphviz)
from sklearn.tree import export_graphviz
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz/bin/'
dot_data = export_graphviz(clf, out_file=None, feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")
graph.view("decision_tree")
