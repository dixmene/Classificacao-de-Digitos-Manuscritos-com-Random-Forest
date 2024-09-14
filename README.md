# **Classificação de Dígitos Manuscritos com Random Forest**

## **Introdução**

Este projeto tem como objetivo desenvolver um sistema para reconhecer dígitos manuscritos, que são imagens de números escritos à mão, utilizando o algoritmo **Random Forest**. O desafio é construir um modelo que possa identificar corretamente cada dígito, de 0 a 9, em imagens pequenas de 8x8 pixels.

## **O Problema**

O reconhecimento de dígitos manuscritos é uma tarefa comum em aprendizado de máquina e visão computacional. O problema é que as imagens podem variar bastante devido a diferentes estilos de escrita. Assim, precisamos de um modelo que seja capaz de lidar com essas variações e classificar cada imagem corretamente.

## **Objetivo**

Nosso objetivo é criar um modelo que possa identificar com precisão os dígitos manuscritos. Para isso, usamos o algoritmo Random Forest, que é conhecido por sua eficácia em tarefas de classificação. Avaliaremos o desempenho do modelo com base em métricas de precisão e analisaremos a matriz de confusão para entender melhor como ele está funcionando.

## **Ferramentas Utilizadas**

- **Linguagem de Programação:** Python
- **Bibliotecas Principais:**
  - **Scikit-learn:** Usada para o algoritmo Random Forest e para avaliação do modelo.
  - **NumPy e Pandas:** Usadas para manipulação de dados.
  - **Matplotlib e Seaborn:** Usadas para visualização dos resultados.


## **Dataset**

- **Fonte:** O dataset `load_digits()` da biblioteca Scikit-learn
- **Características:**
  - **Número de Amostras:** 1.797
  - **Número de Características:** 64 (representando os pixels das imagens 8x8)
  - **Número de Classes:** 10 (dígitos de 0 a 9)

### **Entendimento do Dataset**

O dataset consiste em imagens de 8x8 pixels, cada uma representando um dígito de 0 a 9. Cada imagem é convertida em uma série de 64 valores, onde cada valor representa a intensidade de cor do pixel. Assim, as 64 características de cada amostra são os valores de brilho de cada pixel. O modelo deve aprender a associar esses padrões de pixels aos dígitos correspondentes.

---

## **Processo de Desenvolvimento**

A seguir, descrevo cada etapa do processo de desenvolvimento do modelo, explicando o que estamos fazendo e por que é importante.

### 1. **Preparação dos Dados**

**O que fizemos:** Carregamos o dataset e dividimos os dados em dois grupos: um para treinar o modelo e outro para testar o modelo. Isso nos ajuda a avaliar como o modelo vai se comportar com novos dados.

**Código:**

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Carregar o dataset
digits = load_digits()
X = digits.data
y = digits.target

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Objetivo:**  
Dividir o conjunto de dados para que possamos treinar o modelo em uma parte e testar sua performance na outra. Isso permite avaliar se o modelo está se comportando bem e generalizando adequadamente.

### 2. **Treinamento do Modelo**

**O que fizemos:** Usamos o algoritmo Random Forest para treinar nosso modelo com os dados de treinamento. O Random Forest é uma técnica que combina várias árvores de decisão para melhorar a precisão da classificação.

**Código:**

```python
from sklearn.ensemble import RandomForestClassifier

# Criar e treinar o classificador
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```

**Objetivo:**  
Criar um modelo robusto que possa capturar padrões nos dados de treinamento e usá-los para prever corretamente os dígitos nos dados de teste. Utilizamos o **Random Forest** por sua capacidade de lidar bem com datasets que podem ter ruído ou variações complexas, como é o caso de dígitos manuscritos.
### 3. **Avaliação do Modelo**

**O que fizemos:** Após o treinamento, testamos o modelo com o conjunto de dados de teste e avaliamos seu desempenho. Utilizamos o relatório de classificação para medir a precisão do modelo e a matriz de confusão para visualizar os erros.

**Código:**

```python
from sklearn.metrics import classification_report, confusion_matrix

# Previsões
y_pred = clf.predict(X_test)

# Relatório de Classificação
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
```
### 4. **Importância das Características**

**O que fizemos:** Avaliamos quais pixels (ou características) são mais importantes para a classificação dos dígitos. Isso nos ajuda a entender quais partes das imagens são mais relevantes para o modelo.

**Código:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Obter a importância das características
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Limitar a visualização às 15 características mais importantes
num_features = 15
indices = indices[:num_features]
importances = importances[:num_features]

# Plotar a importância das características
plt.figure(figsize=(10,6))
plt.title("Importância das Características - Random Forest")
plt.bar(range(num_features), importances, color="r", align="center")
plt.xticks(range(num_features), indices, rotation=90)
plt.xlabel('Índice da Característica (Pixel)')
plt.ylabel('Importância')
plt.tight_layout()
plt.show()
```
**Objetivo:**  
Avaliar a qualidade do modelo. Queremos verificar se o modelo é preciso, mas também queremos entender onde ele pode estar cometendo erros. A matriz de confusão nos ajuda a visualizar onde o modelo está confundindo classes (por exemplo, quando ele prevê o dígito 3, mas o verdadeiro era 5).

---

### **Resultados Principais**

1. **Desempenho do Modelo**

   O modelo Random Forest alcançou uma acurácia impressionante de **97%**, indicando que ele consegue reconhecer dígitos manuscritos com um alto nível de precisão.

2. **Classification Report**

   O relatório de classificação fornece uma visão detalhada da performance do modelo para cada classe (dígito) e é crucial para entender as áreas de excelência e possíveis melhorias. Aqui estão os principais indicadores de performance:

   - **Precisão:** A porcentagem de previsões corretas para cada classe.
   - **Recall:** A capacidade do modelo de identificar corretamente todas as instâncias de uma classe específica.
   - **F1-Score:** A média harmônica entre precisão e recall, proporcionando uma única métrica para avaliar o desempenho do modelo.
   - **Suporte:** O número total de amostras para cada classe.

   | Classe | Precisão | Recall | F1-Score | Suporte |
   |--------|----------|--------|----------|---------|
   | 0      | 1.00     | 0.97   | 0.98     | 33      |
   | 1      | 0.97     | 1.00   | 0.98     | 28      |
   | 2      | 1.00     | 1.00   | 1.00     | 33      |
   | 3      | 1.00     | 0.94   | 0.97     | 34      |
   | 4      | 0.98     | 1.00   | 0.99     | 46      |
   | 5      | 0.94     | 0.96   | 0.95     | 47      |
   | 6      | 0.97     | 0.97   | 0.97     | 35      |
   | 7      | 0.97     | 0.97   | 0.97     | 34      |
   | 8      | 0.97     | 0.97   | 0.97     | 30      |
   | 9      | 0.95     | 0.95   | 0.95     | 40      |

   **Análise:** O modelo apresentou desempenho robusto em todas as classes. A maioria das classes obteve um F1-Score superior a 0.95, indicando uma boa combinação de precisão e recall. As menores pontuações foram observadas para as classes 5 e 9, onde o modelo teve dificuldades maiores, possivelmente devido à similaridade visual entre alguns dígitos.

3. **Matriz de Confusão**

   A matriz de confusão fornece uma visão detalhada dos erros de classificação do modelo, revelando quais dígitos foram mais frequentemente confundidos entre si.

   | Verdadeiro \ Predito | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  |
   |----------------------|----|----|----|----|----|----|----|----|----|----|
   | 0                    | 32 |  0 |  0 |  0 |  1 |  0 |  0 |  0 |  0 |  0 |
   | 1                    |  0 | 28 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
   | 2                    |  0 |  0 | 33 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
   | 3                    |  0 |  0 |  0 | 32 |  0 |  1 |  0 |  0 |  1 |  0 |
   | 4                    |  0 |  0 |  0 |  0 | 46 |  0 |  0 |  0 |  0 |  0 |
   | 5                    |  0 |  0 |  0 |  0 |  0 | 45 |  1 |  0 |  0 |  1 |
   | 6                    |  0 |  0 |  0 |  0 |  0 |  1 | 34 |  0 |  0 |  0 |
   | 7                    |  0 |  0 |  0 |  0 |  0 |  0 |  0 | 33 |  0 |  1 |
   | 8                    |  0 |  1 |  0 |  0 |  0 |  0 |  0 |  0 | 29 |  0 |
   | 9                    |  0 |  0 |  0 |  0 |  0 |  1 |  0 |  1 |  0 | 38 |

   **Análise:** A matriz de confusão revela que a maioria dos erros ocorre entre dígitos que são visualmente semelhantes. Por exemplo, o modelo frequentemente confunde os dígitos 3 e 5, o que pode ser atribuído a semelhanças nas formas desses números.


## **Explicação dos Graficos**

1. **Importância das Características**

   - **Gráfico da Importância das Características:** Mostra quais pixels (características) são mais relevantes para a decisão do modelo.
   - **Insight:** Identificamos que certos pixels têm um impacto significativo na classificação dos dígitos. Isso pode indicar que o modelo se baseia mais em áreas específicas das imagens para identificar números.

   ![Importância das Características](https://github.com/user-attachments/assets/380627fe-5e78-4950-a771-c965c52312c7)

2. **Matriz de Confusão**

   - **Gráfico da Matriz de Confusão:** Ilustra os erros de classificação do modelo, detalhando como ele se saiu em cada dígito.
   - **Insight:** A matriz de confusão revela quais dígitos foram mais frequentemente confundidos com outros. Por exemplo, dígitos como 3 e 5 são frequentemente confundidos devido à similaridade visual.

   ![Matriz de Confusão](https://github.com/user-attachments/assets/326b6f95-79e9-4b1b-ad3d-741d74fd4b99)

### **Conclusão**

O projeto demonstrou a eficácia do Random Forest na classificação de dígitos manuscritos. A alta taxa de acerto e a análise das características importantes e dos erros de classificação mostram que o modelo é robusto e eficiente. Este projeto não apenas destaca minha habilidade em implementar e avaliar algoritmos de aprendizado de máquina, mas também minha capacidade de interpretar e comunicar resultados complexos de forma clara e impactante.

