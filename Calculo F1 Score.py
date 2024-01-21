from sklearn.metrics import precision_score, recall_score, f1_score

# Exemplo de rótulos verdadeiros e previsões
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 1, 1]

# Calculando Precision, Recall e F1 Score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
