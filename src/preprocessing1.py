import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Função para mapear o curso para a Área do Conhecimento do ENEM
def mapear_area_conhecimento(nome_curso):
    if pd.isna(nome_curso):
        return "Outros"
    
    curso = str(nome_curso).upper()
    
    # 1. Matemática e suas Tecnologias
    if any(k in curso for k in ['MATEMÁTICA', 'MATEMATICA', 'ESTATÍSTICA', 'ESTATISTICA', 'COMPUTAÇÃO', 'COMPUTACAO', 'CIÊNCIA DA COMPUTAÇÃO', 'SISTEMAS DE INFORMAÇÃO', 'ENGENHARIA']):
        return "Matemática e suas Tecnologias"
    
    # 2. Ciências da Natureza e suas Tecnologias
    elif any(k in curso for k in ['MEDICINA', 'BIOLOGIA', 'FÍSICA', 'FISICA', 'QUÍMICA', 'QUIMICA', 'ENFERMAGEM', 'FARMÁCIA', 'ODONTOLOGIA', 'BIOMEDICINA', 'AGRONOMIA', 'VETERINÁRIA', 'NUTRITION', 'NUTRIÇÃO', 'FISIOTERAPIA']):
        return "Ciências da Natureza e suas Tecnologias"
    
    # 3. Ciências Humanas e suas Tecnologias
    elif any(k in curso for k in ['DIREITO', 'HISTÓRIA', 'HISTORIA', 'GEOGRAFIA', 'FILOSOFIA', 'SOCIOLOGIA', 'PEDAGOGIA', 'PSICOLOGIA', 'ADMINISTRAÇÃO', 'ADMINISTRACAO', 'CIÊNCIAS SOCIAIS', 'ECONOMIA', 'SERVIÇO SOCIAL']):
        return "Ciências Humanas e suas Tecnologias"
    
    # 4. Linguagens, Códigos e suas Tecnologias
    elif any(k in curso for k in ['LETRAS', 'LITERATURA', 'LÍNGUA', 'LINGUA', 'ARTES', 'MÚSICA', 'MUSICA', 'TEATRO', 'CINEMA', 'JORNALISMO', 'COMUNICAÇÃO', 'COMUNICACAO', 'DESIGN', 'EDUCAÇÃO FÍSICA', 'EDUCACAO FISICA', 'TRADUÇÃO']):
        return "Linguagens, Códigos e suas Tecnologias"
    
    else:
        return "Outros"

# Criação automática da pasta de destino
output_dir = os.path.join('data', 'case01')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("data/raw/chamada_regular_sisu_2023_1.csv", encoding='latin-1', sep='|', low_memory=False)
dfCopy = df.copy()

# ==========================================================================================
print(f"\n{'='*45} LIMPEZA E CATEGORIZAÇÃO DA BASE {'='*45}")
# ==========================================================================================

print("\n1. Criando a coluna AREA_CONHECIMENTO e Convertendo as notas")

cols_notas = ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M', 'NOTA_R']

# Mapeando as áreas dos cursos
dfCopy['AREA_CONHECIMENTO'] = dfCopy['NOME_CURSO'].apply(mapear_area_conhecimento)

cols_todas = cols_notas + ['NOME_CURSO', 'AREA_CONHECIMENTO']
dfCopy = dfCopy[cols_todas].copy()

# Conversão de notas para float
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

df_train_full, df_test_full = train_test_split(
    dfCopy, 
    test_size=0.20, 
    random_state=42, 
    shuffle=True
)

scaler_full = StandardScaler()
df_train_full[cols_notas] = scaler_full.fit_transform(df_train_full[cols_notas])
df_test_full[cols_notas] = scaler_full.transform(df_test_full[cols_notas])

df_train_full.to_csv(os.path.join(output_dir, 'train_full.csv'), index=False)
df_test_full.to_csv(os.path.join(output_dir, 'test_full.csv'), index=False)

print(f"Treino Completo: {df_train_full.shape[0]} amostras -> 'data/case01/train_full.csv'")
print(f"Teste Completo:  {df_test_full.shape[0]} amostras -> 'data/case01/test_full.csv'")

# ==========================================================================================
print(f"\n{'='*20} DIVISÃO 2: BASE AMOSTRADA PARA O ALGORITMO 03 (HIERÁRQUICO) {'='*20}")
# ==========================================================================================

SAMPLE_SIZE = 10000
df_sampled = dfCopy.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

df_train_sample, df_test_sample = train_test_split(
    df_sampled, 
    test_size=0.20, 
    random_state=42, 
    shuffle=True
)

scaler_sample = StandardScaler()
df_train_sample[cols_notas] = scaler_sample.fit_transform(df_train_sample[cols_notas])
df_test_sample[cols_notas] = scaler_sample.transform(df_test_sample[cols_notas])

df_train_sample.to_csv(os.path.join(output_dir, 'train_sample_hierarchical.csv'), index=False)
df_test_sample.to_csv(os.path.join(output_dir, 'test_sample_hierarchical.csv'), index=False)

print(f"Treino Amostrado: {df_train_sample.shape[0]} amostras -> 'data/case01/train_sample_hierarchical.csv'")
print(f"Teste Amostrado:  {df_test_sample.shape[0]} amostras -> 'data/case01/test_sample_hierarchical.csv'")