import os
import unicodedata
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Função auxiliar para normalizar textos
def normalizar_texto(texto):
    """
    Remove acentos, converte para maiúsculas e normaliza espaços.
    Isso permite que as regras de categorização funcionem
    independentemente da forma como o curso está escrito.
    """
    if pd.isna(texto):
        return ""

    texto = str(texto).upper().strip()

    texto = unicodedata.normalize('NFKD', texto)
    texto = ''.join(
        c for c in texto
        if not unicodedata.combining(c)
    )

    texto = ' '.join(texto.split())

    return texto


# Função para mapear o curso para a Área do Conhecimento do ENEM
def mapear_area_conhecimento(nome_curso):
    curso = normalizar_texto(nome_curso)

    if not curso:
        return "Outros"

    # ==========================================================
    # 1. Linguagens, Códigos e suas Tecnologias
    # ==========================================================
    palavras_linguagens = [
        'LETRAS', 'LINGUA', 'LINGUISTICA', 'LITERATURA',
        'TRADUCAO', 'INTERPRETACAO', 'LIBRAS',

        'ARTES', 'ARTE', 'ARTES VISUAIS', 'PINTURA',
        'DANCA', 'TEATRO', 'MUSICA', 'CINEMA',
        'AUDIOVISUAL', 'ANIMACAO', 'CENOGRAFIA',
        'DIRECAO DE ARTE', 'DESENHO E PLASTICA',
        'CONSERVACAO E RESTAURACAO', 'CONSERVACAO E RESTAURO',
        'DECORACAO', 'LUTERIA',

        'JORNALISMO', 'COMUNICACAO', 'PUBLICIDADE',
        'PROPAGANDA', 'RADIALISMO', 'TELEVISAO',
        'RELACOES PUBLICAS', 'ESTUDOS DE MIDIA',
        'IMAGEM E SOM', 'SOM IMAGEM E MOVIMENTO',

        'DESIGN', 'DESENHO INDUSTRIAL', 'MODA',
        'EDUCACAO FISICA',

        'PRODUCAO CULTURAL',
        'PRODUCAO E POLITICA CULTURAL',
        'PRODUCAO MULTIMIDIA',
        'PRODUCAO PUBLICITARIA',

        'JOGOS DIGITAIS',
        'MIDIAS DIGITAIS',
        'MIDIAS SOCIAIS DIGITAIS',
        'SISTEMAS E MIDIAS DIGITAIS',
        'MUSICOTERAPIA'
    ]

    if any(palavra in curso for palavra in palavras_linguagens):
        return "Linguagens, Códigos e suas Tecnologias"


    # ==========================================================
    # 2. Ciências Humanas e suas Tecnologias
    # ==========================================================
    palavras_humanas = [
        'HISTORIA', 'GEOGRAFIA', 'FILOSOFIA', 'SOCIOLOGIA',
        'CIENCIAS SOCIAIS', 'CIENCIAS HUMANAS',
        'HUMANIDADES', 'ANTROPOLOGIA',
        'ARQUEOLOGIA', 'ESTUDO DE GENERO',
        'DIVERSIDADE CULTURAL',

        'DIREITO', 'CIENCIAS JURIDICAS',
        'PEDAGOGIA', 'PSICOLOGIA',
        'EDUCACAO ESPECIAL', 'EDUCACAO INFANTIL',
        'EDUCACAO DO CAMPO',

        'ADMINISTRACAO', 'GESTAO', 'ECONOMIA',
        'CIENCIA ECONOMICA', 'CIENCIAS ECONOMICAS',
        'CONTABILIDADE', 'CIENCIAS CONTABEIS',
        'COMERCIO', 'FINANCAS',
        'CONTROLADORIA', 'COOPERATIVISMO',

        'RELACOES INTERNACIONAIS',
        'RELACOES ECONOMICAS INTERNACIONAIS',
        'SERVICO SOCIAL', 'CIENCIA POLITICA',
        'CIENCIAS DO ESTADO',
        'POLITICAS PUBLICAS',

        'TURISMO', 'HOTELARIA', 'EVENTOS',
        'SECRETARIADO', 'SECRETARIADO EXECUTIVO',

        'ARQUIVOLOGIA', 'BIBLIOTECONOMIA',
        'MUSEOLOGIA', 'PATRIMONIO',

        'RECURSOS HUMANOS',
        'GESTAO PUBLICA',
        'GESTAO DE NEGOCIOS',
        'GESTAO COMERCIAL',
        'PROCESSOS GERENCIAIS',

        'DESENVOLVIMENTO REGIONAL',
        'LOGISTICA',
        'MARKETING',
        'NEGOCIOS IMOBILIARIOS',
        'CIENCIAS IMOBILIARIAS',

        'SEGURANCA PUBLICA',
        'URBANISMO',

        'CIENCIAS DA RELIGIAO',
        'CIENCIAS DAS RELIGIOES',
        'CIENCIAS DO CONSUMO',
        'CIENCIAS SOCIOAMBIENTAIS'
    ]

    if any(palavra in curso for palavra in palavras_humanas):
        return "Ciências Humanas e suas Tecnologias"


    # ==========================================================
    # 3. Ciências da Natureza e suas Tecnologias
    # ==========================================================
    palavras_natureza = [
        'BIOLOGIA', 'CIENCIAS BIOLOGICAS', 'BIOLOGICAS',
        'BIOMEDICINA', 'BIOTECNOLOGIA',

        'FISICA', 'QUIMICA',
        'PROCESSOS QUIMICOS',

        'MEDICINA', 'ENFERMAGEM', 'FARMACIA',
        'ODONTOLOGIA', 'FISIOTERAPIA', 'NUTRICAO',
        'VETERINARIA', 'GERONTOLOGIA',
        'OFTALMICA', 'TERAPIA OCUPACIONAL',
        'FONOAUDIOLOGIA', 'ESTETICA',

        'AGRONOMIA', 'AGROECOLOGIA', 'AGROINDUSTRIA',
        'AGRONEGOCIO', 'AQUACULTURA', 'AQUICULTURA',
        'ZOOTECNIA', 'CAFEICULTURA', 'FRUTICULTURA',
        'HORTICULTURA', 'VITICULTURA', 'ENOLOGIA',

        'ALIMENTOS', 'CIENCIA E TECNOLOGIA DE ALIMENTOS',
        'CIENCIA E TECNOLOGIA DE LATICINIOS',
        'LATICINIOS', 'PRODUCAO SUCROALCOOLEIRA',
        'GASTRONOMIA',

        'OCEANOGRAFIA', 'OCEANOLOGIA',
        'CIENCIAS AMBIENTAIS', 'CIENCIA AMBIENTAL',
        'CIENCIAS AGRARIAS', 'CIENCIAS AGRICOLAS',
        'CIENCIAS NATURAIS', 'ECOLOGIA',
        'MEIO AMBIENTE', 'SANEAMENTO AMBIENTAL',
        'TOXICOLOGIA', 'CIENCIAS ATMOSFERICAS',
        'METEOROLOGIA',

        'SAUDE', 'SAUDE COLETIVA',
        'RADIOLOGIA', 'SEGURANCA NO TRABALHO',

        'INTERDISCIPLINAR EM BIOSSISTEMAS'
    ]

    if any(palavra in curso for palavra in palavras_natureza):
        return "Ciências da Natureza e suas Tecnologias"


    # ==========================================================
    # 4. Matemática e suas Tecnologias
    # ==========================================================
    palavras_matematica = [
        'MATEMATICA', 'ESTATISTICA',
        'CIENCIAS ATUARIAIS',

        'COMPUTACAO', 'CIENCIA DA COMPUTACAO',
        'CIENCIA DOS DADOS', 'CIENCIA DE DADOS',
        'SISTEMA DE INFORMACAO',
        'SISTEMAS DE INFORMACAO',
        'ANALISE E DESENVOLVIMENTO DE SISTEMAS',
        'DESENVOLVIMENTO DE SISTEMAS',
        'TECNOLOGIA DA INFORMACAO',
        'TECNOLOGIA EM INFORMACAO',
        'TECNOLOGIA DA INFORMACAO E COMUNICACAO',
        'TECNOLOGIA EM SISTEMAS',
        'SISTEMAS PARA INTERNET',
        'REDES DE COMPUTADORES',
        'SEGURANCA DA INFORMACAO',
        'INTELIGENCIA ARTIFICIAL',
        'SOFTWARE',
        'ENGENHARIA DE SOFTWARE',
        'ENGENHARIA DE COMPUTACAO',
        'TECNOLOGIA EM COMPUTACAO',
        'INFORMATICA',

        'ENGENHARIA',
        'ENGENHARIA ELETRICA',
        'ENGENHARIA ELETRONICA',
        'ENGENHARIA MECANICA',
        'ENGENHARIA MECATRONICA',
        'ENGENHARIA DE PRODUCAO',
        'ENGENHARIA CIVIL',
        'ENGENHARIA METALURGICA',
        'ENGENHARIA DE MATERIAIS',
        'ENGENHARIA DE CONTROLE E AUTOMACAO',
        'ENGENHARIA DE TELECOMUNICACOES',
        'ENGENHARIA NAVAL',
        'ENGENHARIA AERONAUTICA',
        'ENGENHARIA AEROESPACIAL',
        'ENGENHARIA DE TRANSPORTES',

        'CIENCIAS EXATAS',
        'ABI - CIENCIAS EXATAS',
        'ASTRONOMIA',
        'CIENCIA E TECNOLOGIA',
        'CIENCIA DA INFORMACAO',

        'GEOPROCESSAMENTO',
        'CARTOGRAFIA',
        'CONSTRUCAO CIVIL',
        'CONSTRUCAO DE EDIFICIOS',
        'AUTOMACAO INDUSTRIAL',
        'ELETRONICA',
        'ELETROTECNICA',
        'MECANICA',
        'MECATRONICA',
        'METALURGIA',
        'EXPRESSAO GRAFICA',
        'ENERGIAS RENOVAVEIS',
        'ESTRADAS',
        'MANUTENCAO INDUSTRIAL',
        'MINERACAO',
        'NANOTECNOLOGIA',
        'SISTEMAS DE ENERGIA',
        'SISTEMAS DE TELECOMUNICACOES',
        'TELEMATICA',
        'TRANSPORTE TERRESTRE',

        'INTERDISCIPLINAR EM CIENCIA E TECNOLOGIA',
        'INTERDISCIPLINAR EM CIENCIA E TECNOLOGIA DO MAR',
        'INTERDISCIPLINAR EM CIENCIA, TECNOLOGIA E INOVACAO',
        'INTERDISCIPLINAR EM INOVACAO, CIENCIA E TECNOLOGIA',
        'INTERDISCIPLINAR EM ENERGIA E SUSTENTABILIDADE',

        'CIENCIAS EXATAS E TECNOLOGICAS',
        'CIENCIAS EXATAS - BI/LI'
    ]

    if any(palavra in curso for palavra in palavras_matematica):
        return "Matemática e suas Tecnologias"


    # ==========================================================
    # 5. Nomes explícitos das áreas
    # ==========================================================
    if 'LINGUAGENS E SUAS TECNOLOGIAS' in curso:
        return "Linguagens, Códigos e suas Tecnologias"

    if 'CIENCIAS DA NATUREZA E SUAS TECNOLOGIAS' in curso:
        return "Ciências da Natureza e suas Tecnologias"

    if 'CIENCIAS HUMANAS E SOCIAIS E SUAS TECNOLOGIAS' in curso:
        return "Ciências Humanas e suas Tecnologias"

    if 'CIENCIAS HUMANAS E SUAS TECNOLOGIAS' in curso:
        return "Ciências Humanas e suas Tecnologias"

    if 'MATEMATICA E COMPUTACAO E SUAS TECNOLOGIAS' in curso:
        return "Matemática e suas Tecnologias"


    return "Outros"


# Configuração de diretórios
output_dir = os.path.join('data', 'case01')
images_dir = os.path.join('images', 'case01')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)


df = pd.read_csv("data/raw/chamada_regular_sisu_2023_1.csv", encoding='latin-1', sep='|', low_memory=False)

dfCopy = df.copy()

# Limpeza e Mapeamento
cols_notas = ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M', 'NOTA_R']
dfCopy['AREA_CONHECIMENTO'] = dfCopy['NOME_CURSO'].apply(mapear_area_conhecimento)
cols_todas = cols_notas + ['NOME_CURSO', 'AREA_CONHECIMENTO']
dfCopy = dfCopy[cols_todas].copy()

for col in ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M']:
    dfCopy[col] = dfCopy[col].astype(str).str.replace(',', '.')
    dfCopy[col] = pd.to_numeric(dfCopy[col], errors='coerce')

dfCopy['NOTA_R'] = pd.to_numeric(dfCopy['NOTA_R'], errors='coerce')
# Remoção de registros sem alguma das cinco notas
# e preservação do índice original para o Profiling
dfCopy = dfCopy.dropna(
    subset=cols_notas
).copy()

dfCopy['ID_ORIGINAL'] = dfCopy.index

dfCopy = dfCopy.reset_index(drop=True)


# Divisão 1: Base Completa (80/20)
df_train_full, df_test_full = train_test_split(dfCopy, test_size=0.20, random_state=42, shuffle=True)
scaler_full = StandardScaler()
df_train_full[cols_notas] = scaler_full.fit_transform(df_train_full[cols_notas])
df_test_full[cols_notas] = scaler_full.transform(df_test_full[cols_notas])
df_train_full.to_csv(os.path.join(output_dir, 'train_full.csv'), index=False)
df_test_full.to_csv(os.path.join(output_dir, 'test_full.csv'), index=False)

# Divisão 2: Base Amostrada para Algoritmo Hierárquico (N=10.000)
SAMPLE_SIZE = 10000
df_sampled = dfCopy.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
df_train_sample, df_test_sample = train_test_split(df_sampled, test_size=0.20, random_state=42, shuffle=True)
scaler_sample = StandardScaler()
df_train_sample[cols_notas] = scaler_sample.fit_transform(df_train_sample[cols_notas])
df_test_sample[cols_notas] = scaler_sample.transform(df_test_sample[cols_notas])
df_train_sample.to_csv(os.path.join(output_dir, 'train_sample_hierarchical.csv'), index=False)
df_test_sample.to_csv(os.path.join(output_dir, 'test_sample_hierarchical.csv'), index=False)


# Resumo do pré-processamento
print(
    f"Treino Completo Processado:  {df_train_full.shape[0]} amostras"
)

print(
    f"Treino Amostrado Processado: {df_train_sample.shape[0]} amostras"
)

print("\n=== Distribuição das Áreas do Conhecimento ===")
print(
    dfCopy['AREA_CONHECIMENTO']
    .value_counts()
)

print("\n=== Cursos classificados como 'Outros' ===")
print(
    dfCopy.loc[
        dfCopy['AREA_CONHECIMENTO'] == 'Outros',
        'NOME_CURSO'
    ]
    .drop_duplicates()
    .sort_values()
    .to_string(index=False)
)