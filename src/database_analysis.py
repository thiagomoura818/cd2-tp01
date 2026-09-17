import pandas as pd

df = pd.read_csv("data/raw/chamada_regular_sisu_2023_1.csv", encoding='latin-1', sep='|')
dfCopy = df.copy()

# ==========================================================================================
print(f"\n{"="*20} INSPEÇÃO DA BASE {"="*20}")
# ==========================================================================================

print("\n1. Analisando as colunas da base:")

total_linhas = len(dfCopy)

print(f"{'COLUNA':<25} | {'TIPO':<10} | {'NULOS (X%)':<22} | {'VALORES ÚNICOS':<16} | {'EXEMPLO'}")
print("-" * 100)

for coluna in dfCopy.columns:
    tipo = str(dfCopy[coluna].dtype)
    nulos = dfCopy[coluna].isnull().sum()
    percentual_nulos = (nulos / total_linhas) * 100 if total_linhas > 0 else 0
    valores_unicos = dfCopy[coluna].nunique()
    
    if valores_unicos == 1:
        status_variancia = "1 (Constante!)"
    else:
        status_variancia = str(valores_unicos)
        
    exemplo = dfCopy[coluna].iloc[0] if total_linhas > 0 else "Base vazia"
    texto_nulos = f"{nulos} ({percentual_nulos:.1f}%)"

    print(f"{coluna:<25} | {tipo:<10} | {texto_nulos:<22} | {status_variancia:<16} | {exemplo}")

print(f"\nShape da base atual={dfCopy.shape}")