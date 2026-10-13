# Pré-processamento da Questão 2 (Regras de Associação).
import numpy as np
import pandas as pd

# Colunas lidas do CSV (perfil, cota, turno, opção e notas)
COLUNAS_LEITURA = [
    "SEXO", "UF_CANDIDATO", "UF_CAMPUS",
    "TP_COTA", "TIPO_MOD_CONCORRENCIA",
    "TURNO", "OPCAO",
    "NOTA_CANDIDATO",
    "NOTA_L", "NOTA_CH", "NOTA_CN", "NOTA_M", "NOTA_R",
]

# Colunas que viram itens no Apriori (as do enunciado + a nota discretizada)
COLUNAS_APRIORI = [
    "SEXO", "UF_CANDIDATO", "CANDIDATO_LOCAL",
    "TP_COTA", "TIPO_MOD_CONCORRENCIA",
    "TURNO", "OPCAO", "NOTA",
]

ROTULOS_NOTA = ["Nota_Baixa", "Nota_Media", "Nota_Alta"]


def carregar_base(caminho):
    """Lê só as colunas necessárias. TP_COTA tem tipos misturados, então é lida como texto."""
    return pd.read_csv(
        caminho,
        sep="|",
        encoding="latin1",
        decimal=",",
        usecols=COLUNAS_LEITURA,
        dtype={"TP_COTA": str},
    )


def tratar_nulos_e_criar_local(df):
    """Nulos de TP_COTA viram 'Indefinido'; cria CANDIDATO_LOCAL; remove UF_CAMPUS."""
    df = df.copy()
    df["TP_COTA"] = df["TP_COTA"].fillna("Indefinido")
    df["CANDIDATO_LOCAL"] = np.where(df["UF_CANDIDATO"] == df["UF_CAMPUS"],
                                     "Mesma_UF", "Outra_UF")
    return df.drop(columns=["UF_CAMPUS"])


def amostrar(df, n=150_000, random_state=42):
    """Amostra aleatória reprodutível."""
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def discretizar_nota(df_amostra):
    """Divide NOTA_CANDIDATO em tercis (pd.qcut). Devolve o DataFrame com a coluna NOTA e os cortes."""
    df_amostra = df_amostra.copy()
    df_amostra["NOTA"], cortes = pd.qcut(
        df_amostra["NOTA_CANDIDATO"], q=3, labels=ROTULOS_NOTA, retbins=True
    )
    return df_amostra, cortes


def binarizar(df_amostra):
    """One-Hot: cada coluna vira itens 'coluna=valor' com True/False."""
    df_cat = df_amostra[COLUNAS_APRIORI].astype(str)
    return pd.get_dummies(df_cat, prefix_sep="=", dtype=bool)


def preparar_matriz(caminho, n_amostra=150_000):
    """Pipeline completo: CSV -> matriz transacional (e a amostra, para consulta)."""
    df = carregar_base(caminho)
    df = tratar_nulos_e_criar_local(df)
    df_amostra = amostrar(df, n_amostra)
    df_amostra, cortes = discretizar_nota(df_amostra)
    matriz = binarizar(df_amostra)
    return matriz, df_amostra, cortes