import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

# ==========================================================================================
print(f"\n{"="*20} LIMPEZA DA BASE {"="*20}")
# ==========================================================================================

print("\n2. Removendo algumas colunas da base:")

dfReduzida = dfCopy.drop(columns=['ANO', 'EDICAO', 'ETAPA', 'DS_ETAPA', 'PERCENTUAL_BONUS', 'NOME_IES', 'SIGLA_IES', 
                                'NOME_CAMPUS', 'CODIGO_CURSO', 'CPF', 'INSCRICAO_ENEM', 'INSCRITO', 'PESO_L', 'PESO_CH',
                                'PESO_CN', 'PESO_M', 'PESO_R', 'NOTA_MINIMA_L', 'NOTA_MINIMA_CH', 'NOTA_MINIMA_CN',
                                'NOTA_MINIMA_M', 'NOTA_MINIMA_R', 'MEDIA_MINIMA', 'NOTA_L_COM_PESO', 'NOTA_CH_COM_PESO', 
                                'NOTA_CN_COM_PESO', 'NOTA_M_COM_PESO', 'NOTA_R_COM_PESO', 'MUNICIPIO_CANDIDATO'])

print("Colunas com todos os registros de mesmo valor: ANO, EDICAO, ETAPA, DS_ETAPA")
print("Colunas com >94% dos registros nulos ou NaN: PERCENTUAL_BONUS")
print("Colunas com ambiguidade: NOME_IES, SIGLA_IES, NOME_CAMPUS, CODIGO_CURSO, PESO_L, PESO_CH, PESO_CN, PESO_M, PESO_R, " \
"NOTA_MINIMA_L, NOTA_MINIMA_CH, \nNOTA_MINIMA_CN, NOTA_MINIMA_M, NOTA_MINIMA_R, MEDIA_MINIMA, NOTA_L_COM_PESO, NOTA_CH_COM_PESO, NOTA_CN_COM_PESO, " \
"NOTA_M_COM_PESO, NOTA_R_COM_PESO")
print("Colunas com dados pessoais: CPF, INSCRICAO_ENEM, INSCRITO, MUNICIPIO_CANDIDATO")

# NÃO TENHO TANTA CERTEZA!!
dfReduzida.drop(columns=['MOD_CONCORRENCIA', 'QT_VAGAS_CONCORRENCIA', 'CODIGO_IES', 'UF_IES', 'CODIGO_CAMPUS', 'UF_CAMPUS',
                         'MUNICIPIO_CAMPUS', 'NOTA_CANDIDATO', 'DS_PERIODICIDADE'], inplace=True)

print("\n=> Colunas apagadas, mas não tenho tanta certeza: \nMOD_CONCORRENCIA, QT_VAGAS_CONCORRENCIA, CODIGO_IES, UF_IES, CODIGO_CAMPUS, UF_CAMPUS, " \
"MUNICIPIO_CAMPUS, CODIGO_CURSO, NOTA_CANDIDATO, DS_PERIODICIDADE")

print("\nColunas restantes:")
for coluna in dfReduzida.columns:
    print(f"{coluna}", end=" ")
print()

# =====
# print("\n3. Tratando valores nulos de TP_COTA (Substituindo NaN por 'NAO')")
# dfReduzida['TP_COTA'] = dfReduzida['TP_COTA'].fillna('NAO')
dfReduzida.drop(columns=['TP_COTA'], inplace=True)

# ==========================================================================================
print(f"\n{"="*20} ENGENHARIA DE FEATURES {"="*20}")
# ==========================================================================================

print("\n4. Mapeamento numérico (CODIGO) para as UF:")

mapa_uf = {
    'AC': 1, 'AL': 2, 'AM': 3, 'AP': 4, 'BA': 5, 'CE': 6, 'DF': 7, 'ES': 8, 'GO': 9,
    'MA': 10, 'MG': 11, 'MS': 12, 'MT': 13, 'PA': 14, 'PB': 15, 'PE': 16, 'PI': 17,
    'PR': 18, 'RJ': 19, 'RN': 20, 'RO': 21, 'RR': 22, 'RS': 23, 'SC': 24, 'SE': 25,
    'SP': 26, 'TO': 27
}

colunas_uf = ['UF_IES', 'UF_CAMPUS', 'UF_CANDIDATO']

for coluna in colunas_uf:
    if coluna in dfReduzida.columns:
        dfReduzida[coluna] = dfReduzida[coluna].map(mapa_uf)

print("Colunas mapeadas: UF_IES, UF_CAMPUS, UF_CANDIDADO")

# =====
print("\n5. Binarizando colunas com apenas dois valores distintos:")

dfReduzida['OPCAO'] = dfReduzida['OPCAO'] - 1
print("Coluna OPCAO: 1 -> 0 e 2 -> 1")

dfReduzida['APROVADO'] = dfReduzida['APROVADO'].map({'N': 0, 'S': 1})
print("Coluna APROVADO: N -> 0 e S -> 1")

dfReduzida['SEXO'] = dfReduzida['SEXO'].map({'F': 0, 'M': 1})
print("Coluna SEXO: F -> 0 e M -> 1")

dfReduzida['MATRICULA'] = dfReduzida['MATRICULA'].apply(lambda x: 1 if x == 'EFETIVADA' else 0)
print("Coluna MATRICULA: EFETIVADA -> 1 e outros -> 0")

# =====
print("\n6. Renomeando coluna DT_NASCIMENTO para IDADE e realizando os cálculos")
dfReduzida.rename(columns={'DT_NASCIMENTO': 'IDADE'}, inplace=True)
dfReduzida['IDADE'] = 2026 - dfReduzida['IDADE']

# =====
print("\n7. Convertendo as colunas de NOTA_* para o tipo float")

colunas_notas = ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M', 'NOTA_R', 'NOTA_CORTE']

for coluna in colunas_notas:
    if coluna in dfReduzida.columns:
        dfReduzida[coluna] = dfReduzida[coluna].astype(str).str.replace(',', '.', regex=False)
        dfReduzida[coluna] = pd.to_numeric(dfReduzida[coluna], errors='coerce')


# =====
print("\n8. Criando a coluna MEDIA_NOTAS do candidado")

colunas_notas = ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M', 'NOTA_R']
dfReduzida['MEDIA_NOTAS'] = dfReduzida[colunas_notas].mean(axis=1)

# =====
print("\n9. Discretizando a coluna MEDIA_NOTAS nas colunas NOTA_BAIXA, NOTA_MEDIA, NOTA_ALTA")

intervalos = [0, 333, 666, 1000]
nomes_faixas = ['NOTA_BAIXA', 'NOTA_MEDIA', 'NOTA_ALTA']

faixas_notas = pd.cut(dfReduzida['MEDIA_NOTAS'], bins=intervalos, labels=nomes_faixas, include_lowest=True)
df_dummies = pd.get_dummies(faixas_notas, dtype=int)
dfReduzida = pd.concat([dfReduzida, df_dummies], axis=1)

# =====
print("\n10. Aplicando One-Hot Encoding:")

df_dummies = pd.get_dummies(
    # dfReduzida[['GRAU', 'TURNO', 'TP_COTA', 'TIPO_MOD_CONCORRENCIA']], 
    dfReduzida[['GRAU', 'TURNO','TIPO_MOD_CONCORRENCIA']], 
    dtype=int
)

mapa_abreviacoes = {
    'GRAU_Bacharelado': 'GRAU_BACHAREL',
    'GRAU_Licenciatura': 'GRAU_LICENC',
    'GRAU_Tecnológico': 'GRAU_TEC',
    'GRAU_Área Básica de Ingresso (ABI)': 'GRAU_ABI',
    
    'TURNO_Integral': 'TURNO_INT',
    'TURNO_Noturno': 'TURNO_NOT',
    'TURNO_Matutino': 'TURNO_MAT',
    'TURNO_Vespertino': 'TURNO_VESP',
    'TURNO_EaD': 'TURNO_EAD',
    
    # 'TP_COTA_PPI': 'COTA_PPI',
    # 'TP_COTA_D': 'COTA_D',
    # 'TP_COTA_PP': 'COTA_PP',
    # 'TP_COTA_PPID': 'COTA_PPID',
    # 'TP_COTA_DD': 'COTA_DD',
    # 'TP_COTA_I': 'COTA_I',
    # 'TP_COTA_PPD': 'COTA_PPD',
    
    'TIPO_MOD_CONCORRENCIA_L': 'CONC_L',
    'TIPO_MOD_CONCORRENCIA_A': 'CONC_A',
    'TIPO_MOD_CONCORRENCIA_B': 'CONC_B',
    'TIPO_MOD_CONCORRENCIA_V': 'CONC_V'
}

df_dummies.rename(columns=mapa_abreviacoes, inplace=True)
dfReduzida = pd.concat([dfReduzida, df_dummies], axis=1)

# dfReduzida.drop(columns=['GRAU', 'TURNO', 'TP_COTA', 'TIPO_MOD_CONCORRENCIA'], inplace=True)
dfReduzida.drop(columns=['GRAU', 'TURNO', 'TIPO_MOD_CONCORRENCIA'], inplace=True)
print("Colunas tratadas e deletadas: GRAU, TURNO, TP_COTA, TIPO_MOD_CONCORRENCIA")

# =====
print("\n11. Aplicando StandardScaler nas colunas de notas")

scaler = StandardScaler()
colunas_para_escalar = ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M', 'NOTA_R', 'NOTA_CORTE', 'MEDIA_NOTAS']
colunas_validas = [col for col in colunas_para_escalar if col in dfReduzida.columns]

if colunas_validas:
    for col in colunas_validas:
        dfReduzida[col] = dfReduzida[col].fillna(dfReduzida[col].mean())
    dfReduzida[colunas_validas] = scaler.fit_transform(dfReduzida[colunas_validas])

# =====
# TESTES - CONSULTAS RAPIDAS - CURIOSIDADE

print("\nPrimeiras linhas:")
print(dfReduzida.head())

print("\n3. Existem de linhas duplicadas?")
duplicates = dfReduzida.duplicated().sum()
print(f"Total de linhas duplicadas: {duplicates}")

print(f"\nShape da base atual={dfReduzida.shape}")

# =============== MATRIZ DE CORRELAÇÃO DE PEARSON ========================


df_numerico = dfReduzida.select_dtypes(include=['number'])

matriz_corr = df_numerico.corr(method='pearson')

tamanho_janela = max(10, len(df_numerico.columns) * 0.6)
plt.figure(figsize=(tamanho_janela, tamanho_janela * 0.8))

# 4. Desenha o mapa de calor (Heatmap)
# cmap='coolwarm': tons azuis para correlação negativa, vermelhos para positiva
# annot=True: exibe os valores numéricos dentro dos quadrados
# fmt=".2f": limita os valores a 2 casas decimais
sns.heatmap(
    matriz_corr, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1, 
    linewidths=0.5,
    cbar_kws={"shrink": .8}
)

# 5. Ajusta os rótulos para não cortarem na imagem
plt.title('Matriz de Correlação de Pearson', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

# 6. Salva o gráfico em formato PNG de alta resolução (300 DPI)
plt.savefig('matriz_correlacao.png', dpi=300, bbox_inches='tight')

plt.close()

# ========================= CONC_L ===================================

# 1. Calcula as frequências absolutas (contagem) e relativas (percentual)

colunas_alvo = ['CONC_L', 'CONC_A', 'CONC_B', 'CONC_V']

for coluna in colunas_alvo:
    if coluna not in dfReduzida: 
        print(f"Coluna {coluna} nao existe!")
    else: 
        contagem = dfReduzida[coluna].value_counts(dropna=False)
        percentual = dfReduzida[coluna].value_counts(normalize=True, dropna=False) * 100

        # 2. Exibe o cabeçalho organizado
        print(f"=== DISTRIBUIÇÃO DA COLUNA: {coluna} ===")
        print(f"{'VALOR':<10} | {'CONTAGEM':<12} | {'PERCENTUAL'}")
        print("-" * 40)

        # 3. Varre os valores distintos encontrados e imprime formatado
        for valor, qtd in contagem.items():
            pct = percentual[valor]
            # Se o valor for nulo, exibe como 'NaN' para facilitar a leitura
            nome_valor = "NaN" if pd.isna(valor) else str(valor)
            print(f"{nome_valor:<10} | {qtd:<12,} | {pct:.2f}%")
