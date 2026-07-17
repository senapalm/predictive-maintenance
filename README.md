# Predictive Maintenance

Projeto avaliativo desenvolvido no contexto do Carreira Tech - Trilha de Inteligência Articial (Fundamentos de Dados, Programação e Análise Preditiva com Python), com o objetivo de construir um modelo de Machine Learning capaz de prever falhas em máquinas industriais a partir de dados operacionais.

## Link para o video explicativo:
---

## Objetivo

A manutenção preditiva busca identificar equipamentos com maior probabilidade de falha antes que o problema ocorra, permitindo reduzir custos de manutenção, evitar paradas inesperadas e aumentar a disponibilidade dos equipamentos.

Neste projeto foram aplicadas técnicas de Ciência de Dados e Machine Learning para comparar diferentes algoritmos de classificação e selecionar o modelo com melhor desempenho.

---

## Dataset

O projeto utiliza um conjunto de dados contendo informações de operação de máquinas industriais, incluindo variáveis como:

- Temperatura do ar
- Temperatura do processo
- Velocidade de rotação
- Torque
- Desgaste da ferramenta
- Tipo da máquina

Variável alvo:

- **Falha da máquina**

---

## Estrutura do Projeto

```
predictive-maintenance/
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── projeto_final_sena.ipynb
│
├── src/
│   ├── phase00_data.py
│   ├── phase01_eda.py
│   ├── phase02_preprocessing.py
│   ├── phase03_featengineering.py
│   ├── phase04_balancing.py
│   ├── phase05_scaling.py
│   ├── phase06_decision_tree.py
│   └── phase07_evaluation.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Fluxo do Projeto

O desenvolvimento foi dividido em sete etapas:

### Fase 00 – Data Acquisition

- Download do dataset
- Carregamento dos dados

### Fase 01 – Exploratory Data Analysis (EDA)

- Estatísticas descritivas
- Histogramas
- Scatter Matrix
- Balanceamento das classes

### Fase 02 – Data Preprocessing

- Remoção de duplicatas
- Tratamento de valores ausentes
- Conversão de variáveis booleanas

### Fase 03 – Feature Engineering

Criação de novas variáveis:

- Potência (RPM × Torque)
- Diferença entre temperaturas
- One-Hot Encoding da variável Tipo

### Fase 04 – Data Balancing

- Separação entre Features e Target
- Balanceamento utilizando SMOTE

### Fase 05 – Feature Scaling

- Aplicação do StandardScaler para o modelo KNN
- Dados da Árvore de Decisão mantidos sem escalonamento

### Fase 06 – Modelagem

Treinamento e otimização de:

- K-Nearest Neighbors (KNN)
- Decision Tree

### Fase 07 – Avaliação

Comparação entre os modelos utilizando:

- Accuracy no conjunto de teste
- Tabela comparativa dos resultados
- Seleção do modelo com melhor desempenho
- Conclusão técnica sobre o modelo recomendadoS

---

## Tecnologias Utilizadas

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Requests
- Jupyter Notebook

---

## Como executar

### Clone o projeto

```bash
git clone https://github.com/senapalm/predictive-maintenance.git
```

### Entre na pasta

```bash
cd predictive-maintenance
```

### Crie um ambiente virtual

Windows

```bash
python -m venv .venv
```

Linux/macOS

```bash
python3 -m venv .venv
```

### Ative o ambiente virtual

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

### Instale as dependências

```bash
pip install -r requirements.txt
```

### Execute o notebook

```bash
jupyter notebook
```

Abra o arquivo:

```
notebooks/projeto_final_sena.ipynb
```
---

## Resultados

Após o treinamento dos modelos, foram comparados os algoritmos KNN e Árvore de Decisão utilizando o conjunto de teste.

O modelo com melhor desempenho foi selecionado como solução final para o problema de manutenção preditiva.

---

## Melhorias Futuras (Segundo sugestões de especialistas)

- Avaliação com Random Forest
- Avaliação com XGBoost
- Ajuste de hiperparâmetros com GridSearchCV
- Pipeline completo utilizando Scikit-Learn
- Deploy em aplicação web utilizando Streamlit
- Integração com dados em tempo real

---

## Autor

**Alexandre Sena**