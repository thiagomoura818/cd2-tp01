# Ciência de Dados II - Trabalho Prático 01

**Disciplina:** Ciência de Dados II (G07CDAD2.01) - CEFET-MG  
**Professor:** Thiago de Sousa Goveia  
**Tema:** Agrupamento, Associação e Classificação com MLP  

Este projeto objetiva integrar métodos de aprendizado não supervisionado (Agrupamento e Associação) com redes neurais profundas (MLP utilizando TensorFlow/Keras) para extrair padrões e inteligência de dados sobre o comportamento de candidatos nas universidades públicas (SiSU 2023.1).

---

## 📁 Arquitetura do Projeto (Organização de Pastas)

O projeto foi estruturado buscando organização e escalabilidade, dividindo as etapas de experimentação (notebooks) e código reutilizável (src):

```
/
├── data/
│   ├── raw/             # Coloque aqui os dados originais brutos (chamada_regular_sisu_2023_1.csv)
│   └── processed/       # Bases de dados pós-limpeza, discretização ou binarização
├── notebooks/
│   ├── 01_agrupamento.ipynb        # Modelagem de perfis com K-Means, DBSCAN e Hierárquico
│   ├── 02_regras_associacao.ipynb  # Regras de Associação com Apriori (mlxtend)
│   └── 03_classificacao_mlp.ipynb  # Predição do Status de Matrícula (TensorFlow/Keras)
├── reports/
│   └── figures/         # Gráficos estáticos gerados (ex: Dendrograma, Curvas Cotovelo/K-dist)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py # Scripts utilitários de limpeza e One-Hot Encoding
│   ├── clustering.py    # Códigos para agrupamento e cálculo de silhueta
│   ├── association.py   # Utilitários de discretização para Apriori
│   └── classification.py# Arquitetura das redes MLP, baseline e relatórios (Matriz Confusão, etc.)
├── main.py              # Ponto de entrada (opcional) para rodar os dados do projeto
├── README.md            # Tutorial e informações gerais do projeto
└── requirements.txt     # Dependências do projeto
```

---

## ⚙️ Instalação (Pré-requisitos)

Para executar os códigos (seja os notebooks ou os módulos em python), é recomendada a criação de um ambiente virtual para que não existam conflitos de bibliotecas na sua máquina local com as dependências do TensorFlow 2.x e afins.

1. **Abra o terminal na pasta raiz do projeto (`cd02/tp01/`):**

2. **Crie um ambiente virtual (Opcional, mas recomendado):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Para Linux/Mac
   # Para Windows: venv\Scripts\activate
   ```

3. **Instale as dependências via pip usando o `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Como Utilizar o Projeto

### 1. Coloque os Dados
Antes de rodar os algoritmos, obtenha o arquivo [`chamada_regular_sisu_2023_1.csv`](https://drive.google.com/file/d/14Vm750yRvZSvPGf_ZOw3YYgH5MvNFfs9/view?usp=sharing) e coloque-o **obrigatoriamente** dentro da pasta `data/raw/`. 

- Para uma inspeção da base de dados original, sem manipulações, execute o arquivo `src\database_analysis.py`.
- Para reproduzir a limpeza dos dados, tratamentos e divisão em conjunto de treino e teste, execute os arquivos `src\preprocessingID.py`. No qual ID pode ser ou 1 ou 2 ou 3, de acordo com os casos abordados no trabalho. As divisões de base resultante de cada arquivo de pre-processamento são salvar automaticamente dentro da respectiva pasta `data/case0ID/`.

### 2. Rodando os Jupyter Notebooks
Como a especificação do trabalho pede que o arquivo submetido seja `.ipynb` executado de ponta a ponta, os notebooks foram segmentados por domínio. Para abrir:

```bash
# Com o seu venv ativo:
jupyter notebook
```
Acesse a pasta `notebooks/` pelo seu navegador e execute, na ordem:
- `01_agrupamento.ipynb`
- `02_regras_associacao.ipynb` 
- `03_classificacao_mlp.ipynb`

### 3. Garantindo a Reprodutibilidade
Conforme especificação, todas as instâncias que utilizam aleatoriedade (padronização, quebras em treino/teste e inicialização de pesos nas MLPs) precisam manter uma mesma *seed*. No projeto, as sementes foram fixadas como **42** (`random_state = 42` e `tf.random.set_seed(42)`).

---
