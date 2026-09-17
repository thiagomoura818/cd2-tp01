import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Criação automática da pasta de destino
output_dir = os.path.join('data', 'case01')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("data/raw/chamada_regular_sisu_2023_1.csv", encoding='latin-1', sep='|')
dfCopy = df.copy()

# ==========================================================================================
print(f"\n{'='*45} LIMPEZA DA BASE {'='*45}")
# ==========================================================================================

print("\n1. Convertendo as notas para o tipo FLOAT64")

cols_notas = ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M', 'NOTA_R']
cols_todas = cols_notas + ['NOME_CURSO']

dfCopy = dfCopy[cols_todas].copy()

for col in ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M']:
    dfCopy[col] = dfCopy[col].astype(str).str.replace(',', '.')
    dfCopy[col] = pd.to_numeric(dfCopy[col], errors='coerce')

dfCopy['NOTA_R'] = pd.to_numeric(dfCopy['NOTA_R'], errors='coerce')

# =====
print("\n2. Removendo registros com dados ausentes ou inválidos nas notas")

dfCopy = dfCopy.dropna(subset=cols_notas).reset_index(drop=True)

# ==========================================================================================
print(f"\n{'='*35} DIVISÃO 1: BASE COMPLETA (80/20) {'='*35}")
# ==========================================================================================

print("\n3. Divisão da base completa em treino e teste (80/20)")

df_train_full, df_test_full = train_test_split(
    dfCopy, 
    test_size=0.20, 
    random_state=42, 
    shuffle=True
)

print("Aplicando o StandardScaler em notas")

scaler_full = StandardScaler()
df_train_full[cols_notas] = scaler_full.fit_transform(df_train_full[cols_notas])
df_test_full[cols_notas] = scaler_full.transform(df_test_full[cols_notas])

print("Salvando em arquivos CSV dentro de 'data/case01'\n")

df_train_full.to_csv(os.path.join(output_dir, 'train_full.csv'), index=False)
df_test_full.to_csv(os.path.join(output_dir, 'test_full.csv'), index=False)

print(f"Treino Completo: {df_train_full.shape[0]} amostras -> 'data/case01/train_full.csv'")
print(f"Teste Completo:  {df_test_full.shape[0]} amostras -> 'data/case01/test_full.csv'")

print("\nTreino Completo:")
print(df_train_full.head())

print("\nTeste Completo:")
print(df_test_full.head())

# ==========================================================================================
print(f"\n{'='*20} DIVISÃO 2: BASE AMOSTRADA PARA O ALGORITMO 03 (HIERÁRQUICO) {'='*20}")
# ==========================================================================================

print("\n4. Divisão da base amostrada em treino e teste (80/20)")

SAMPLE_SIZE = 10000
print(f"Tamanho da amostra: {SAMPLE_SIZE}")

df_sampled = dfCopy.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

df_train_sample, df_test_sample = train_test_split(
    df_sampled, 
    test_size=0.20, 
    random_state=42, 
    shuffle=True
)

print("Aplicando o StandardScaler em notas da amostragem")

scaler_sample = StandardScaler()
df_train_sample[cols_notas] = scaler_sample.fit_transform(df_train_sample[cols_notas])
df_test_sample[cols_notas] = scaler_sample.transform(df_test_sample[cols_notas])

print("Salvando em arquivos CSV dentro de 'data/case01'\n")

df_train_sample.to_csv(os.path.join(output_dir, 'train_sample_hierarchical.csv'), index=False)
df_test_sample.to_csv(os.path.join(output_dir, 'test_sample_hierarchical.csv'), index=False)

print(f"Treino Amostrado: {df_train_sample.shape[0]} amostras -> 'data/case01/train_sample_hierarchical.csv'")
print(f"Teste Amostrado:  {df_test_sample.shape[0]} amostras -> 'data/case01/test_sample_hierarchical.csv'")

print("\nTreino Amostrado:")
print(df_train_sample.head())

print("\nTeste Amostrado:")
print(df_test_sample.head())