import numpy as np
import pandas as pd

df = pd.read_csv("data/raw/chamada_regular_sisu_2023_1.csv", encoding='latin-1', sep='|')
dfCopy = df.copy()

# ==========================================================================================
print(f"\n{"="*20} INSPEÇÃO DA BASE {"="*20}")
# ==========================================================================================

print("\nPrimeiras linhas:")
print(dfCopy.head())

print("\n1. Analisando as colunas da base:")

total_linhas = len(dfCopy)

# Cabeçalho formatado para organizar a leitura
print(f"{'COLUNA':<25} | {'TIPO':<10} | {'NULOS (X%)':<22} | {'VALORES ÚNICOS':<16} | {'EXEMPLO'}")
print("-" * 100)

for coluna in dfCopy.columns:
    tipo = str(dfCopy[coluna].dtype)
    nulos = dfCopy[coluna].isnull().sum()
    percentual_nulos = (nulos / total_linhas) * 100 if total_linhas > 0 else 0
    
    # Identifica a quantidade de valores únicos (ignora nulos por padrão)
    valores_unicos = dfCopy[coluna].nunique()
    
    # Alerta se a coluna tiver o mesmo valor sempre (1 único valor)
    if valores_unicos == 1:
        status_variancia = "1 (Constante!)"
    else:
        status_variancia = str(valores_unicos)
        
    exemplo = dfCopy[coluna].iloc[0] if total_linhas > 0 else "Base vazia"
    
    # Texto formatado para os nulos
    texto_nulos = f"{nulos} ({percentual_nulos:.1f}%)"
    
    # Impressão com espaçamento fixo alinhado à esquerda (<)
    print(f"{coluna:<25} | {tipo:<10} | {texto_nulos:<22} | {status_variancia:<16} | {exemplo}")

# =====
print("\n2. Removendo algumas colunas da base:")

dfReduzida = dfCopy.drop(columns=['ANO', 'EDICAO', 'ETAPA', 'DS_ETAPA', 'PERCENTUAL_BONUS', 'NOME_IES', 'SIGLA_IES', 
                                'NOME_CAMPUS', 'NOME_CURSO', 'CPF', 'INSCRICAO_ENEM', 'INSCRITO', 'PESO_L', 'PESO_CH',
                                'PESO_CN', 'PESO_M', 'PESO_R', 'NOTA_MINIMA_L', 'NOTA_MINIMA_CH', 'NOTA_MINIMA_CN',
                                'NOTA_MINIMA_M', 'NOTA_MINIMA_R', 'MEDIA_MINIMA', 'NOTA_L_COM_PESO', 'NOTA_CH_COM_PESO', 
                                'NOTA_CN_COM_PESO', 'NOTA_M_COM_PESO', 'NOTA_R_COM_PESO', 'NOTA_L', 'NOTA_CH', 'NOTA_CN', 
                                'NOTA_M', 'NOTA_R'])

print("Colunas com todos os registros de mesmo valor: ANO, EDICAO, ETAPA, DS_ETAPA")
print("Colunas com >94% dos registros nulos ou NaN: PERCENTUAL_BONUS")
print("Colunas com ambiguidade: NOME_IES, SIGLA_IES, NOME_CAMPUS, NOME_CURSO, PESO_L, PESO_CH, PESO_CN, PESO_M, PESO_R, " \
"NOTA_MINIMA_L, NOTA_MINIMA_CH, \nNOTA_MINIMA_CN, NOTA_MINIMA_M, NOTA_MINIMA_R, MEDIA_MINIMA, NOTA_L_COM_PESO, NOTA_CH_COM_PESO, NOTA_CN_COM_PESO, " \
"NOTA_M_COM_PESO, NOTA_R_COM_PESO, \nNOTA_L, NOTA_CH, NOTA_CN, NOTA_M, NOTA_R")
print("Colunas com dados pessoais: CPF, INSCRICAO_ENEM, INSCRITO")

print("\nPrimeiras linhas:")
print(dfReduzida.head())

# =====
print("\n3. Existência de linhas duplicadas")
duplicates = dfReduzida.duplicated().sum()
print(f"Total de linhas duplicadas: {duplicates}")

