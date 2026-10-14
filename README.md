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

## Decisões principais por questão

**Associação (notebook 02)**
- Amostra de 150 mil linhas (`random_state=42`): a base inteira não cabe no Apriori; a amostra preserva as proporções (diferenças de até 0,002).
- Nulos de `TP_COTA` (54,7%) viram `"Indefinido"`: o nulo ocorre exatamente quando a modalidade não é `L` (Lei de Cotas).
- `CANDIDATO_LOCAL` = `Mesma_UF` se `UF_CANDIDATO == UF_CAMPUS`, senão `Outra_UF`.
- Nota em 3 categorias por **tercis** (`pd.qcut`, cortes em 539,58 e 617,66): grupos de mesmo tamanho. Com `pd.cut` (largura igual) ficaria 0,4% / 83,2% / 16,4%.
- Suporte mínimo 0,01 (4.123 itemsets) e confiança mínima 0,6 (6.188 regras), escolhidos por tabelas de calibração.
- Filtros: Lift > 1; sem regras que usam `TP_COTA` e `TIPO_MOD_CONCORRENCIA` juntas; sem redundantes (ganho mínimo de 0,02 de confiança sobre a regra mais simples, antecedente de até 3 itens); sem a modalidade `B` (bonificação regional); uma regra por itemset. Resultado final: 187 regras.

**Classificação (notebook 03)**
- Apenas aprovados (`APROVADO == 'S'`) com `MATRICULA` em `EFETIVADA` ou `NÃO COMPARECEU` (205.530 candidatos, classes ~50/50).
- Divisão estratificada 70/15/15; imputação, padronização (`StandardScaler`) e One-Hot ajustados só no treino (sem vazamento).
- Atributos criados: idade, margem de aprovação, razão classificação/vagas, mesma UF, distância candidato-campus (Haversine com coordenadas do IBGE) e estatísticas das notas.
- Corte de decisão pelo índice de Youden na validação; melhor experimento escolhido pela AUC de validação; teste só para relatar.

**Agrupamento (notebook 01)**
- Somente as 5 notas brutas, padronizadas com `StandardScaler`; perfilamento na escala original.

---

## Limitações conhecidas

- A base tem uma linha por candidatura: o mesmo candidato pode aparecer duas vezes (opção 1 e 2). As regras e os modelos descrevem candidaturas, não pessoas.
- Regras de associação mostram co-ocorrência, não causalidade.
- A distância do notebook 03 usa o município (não o endereço) e depende de nomes que casam com a base do IBGE (baixada de um CSV público no GitHub; precisa de internet na primeira execução).