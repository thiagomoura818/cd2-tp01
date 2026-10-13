# Apriori e filtros de regras da Regras de Associação
import time
from itertools import combinations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Parâmetros escolhidos na calibração (seções 6 a 8 do notebook)
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.6
MIN_MELHORIA = 0.02     # a regra complexa precisa ganhar 2 p.p. de confiança
MAX_ANTECEDENTE = 3     # no máximo 3 itens no "se"


def calibrar_suporte(matriz, suportes=(0.10, 0.05, 0.02, 0.01)):
    """Roda o Apriori para cada suporte mínimo. Devolve a tabela de calibração e os itemsets de cada rodada."""
    resultados, itemsets_por_suporte = [], {}
    for ms in suportes:
        inicio = time.time()
        itens = apriori(matriz, min_support=ms, use_colnames=True)
        tempo = time.time() - inicio
        resultados.append({
            "min_support": ms,
            "n_itemsets": len(itens),
            "maior_itemset": itens["itemsets"].apply(len).max(),
            "tempo_s": round(tempo, 1),
        })
        itemsets_por_suporte[ms] = itens
    return pd.DataFrame(resultados), itemsets_por_suporte


def gerar_regras(itens, n_transacoes, conf_min):
    """association_rules compatível com versões novas e antigas do mlxtend."""
    try:
        return association_rules(itens, num_itemsets=n_transacoes,
                                 metric="confidence", min_threshold=conf_min)
    except TypeError:  # versões antigas não têm num_itemsets
        return association_rules(itens, metric="confidence", min_threshold=conf_min)


def calibrar_confianca(itemsets, n_transacoes, confiancas=(0.5, 0.6, 0.7)):
    """Gera as regras para cada confiança mínima. Devolve a tabela de calibração e as regras de cada rodada."""
    resultados, regras_por_conf = [], {}
    for conf in confiancas:
        inicio = time.time()
        regras = gerar_regras(itemsets, n_transacoes, conf)
        tempo = time.time() - inicio
        resultados.append({
            "min_confidence": conf,
            "n_regras": len(regras),
            "n_com_lift>1": (regras["lift"] > 1).sum(),
            "lift_maximo": round(regras["lift"].max(), 2),
            "tempo_s": round(tempo, 1),
        })
        regras_por_conf[conf] = regras
    return pd.DataFrame(resultados), regras_por_conf


def colunas_do_conjunto(conjunto):
    """Colunas de origem dos itens ('TP_COTA=PPI' -> 'TP_COTA')."""
    return {item.split("=")[0] for item in conjunto}


def texto(conjunto):
    """Deixa o conjunto legível: 'A, B'."""
    return ", ".join(sorted(conjunto))


def filtrar_lift_e_triviais(regras):
    """Lift > 1 e remove regras que usam TP_COTA e TIPO_MOD_CONCORRENCIA ao mesmo tempo.
    Devolve as regras com a coluna 'itemset' (antecedente + consequente)."""
    regras = regras[regras["lift"] > 1].copy()
    par_redundante = {"TP_COTA", "TIPO_MOD_CONCORRENCIA"}
    usa_as_duas = regras.apply(
        lambda r: par_redundante <= colunas_do_conjunto(r["antecedents"] | r["consequents"]),
        axis=1,
    )
    regras = regras[~usa_as_duas].copy()
    regras["itemset"] = regras.apply(lambda r: r["antecedents"] | r["consequents"], axis=1)
    return regras


def uma_regra_por_itemset(regras):
    """O lift é simétrico, então fica só a regra de maior confiança de cada itemset (ordenada por lift)."""
    melhor = regras.groupby("itemset")["confidence"].idxmax()
    return regras.loc[melhor].sort_values("lift", ascending=False).reset_index(drop=True)


def filtrar_redundantes(regras, itemsets, min_melhoria=MIN_MELHORIA, max_antecedente=MAX_ANTECEDENTE):
    """Mantém a regra só se superar a melhor regra mais simples (mesmo consequente) em min_melhoria
    de confiança, e limita o tamanho do antecedente. Adiciona as colunas 'melhoria' e 'tam_ant'."""
    # itemset -> suporte (os subconjuntos de itemsets frequentes também estão aqui)
    suporte = dict(zip(itemsets["itemsets"], itemsets["support"]))

    def melhoria(regra):
        ant, con = regra["antecedents"], regra["consequents"]
        melhor_simples = 0.0
        for k in range(1, len(ant)):                 # subconjuntos próprios e não vazios
            for sub in combinations(ant, k):
                sub = frozenset(sub)
                melhor_simples = max(melhor_simples, suporte[sub | con] / suporte[sub])
        return regra["confidence"] - melhor_simples  # antecedente de 1 item: melhoria = confiança

    regras = regras.copy()
    regras["melhoria"] = regras.apply(melhoria, axis=1)
    regras["tam_ant"] = regras["antecedents"].apply(len)
    sem_red = regras[regras["melhoria"] >= min_melhoria]
    return sem_red[sem_red["tam_ant"] <= max_antecedente].copy()


def remover_modalidade_b(regras):
    """Remove regras com a modalidade B (bonificação regional): refletem o desenho da política."""
    return regras[~regras["itemset"].apply(lambda s: "TIPO_MOD_CONCORRENCIA=B" in s)]


def top_regras(regras, n=5):
    """Top-n por lift, com colunas de texto para leitura."""
    top = regras.head(n).copy()
    top["antecedente"] = top["antecedents"].apply(texto)
    top["consequente"] = top["consequents"].apply(texto)
    return top[["antecedente", "consequente", "support", "confidence", "lift"]].round(3)


def executar_associacao(matriz):
    """Pipeline completo a partir da matriz transacional, com os parâmetros escolhidos."""
    itemsets = apriori(matriz, min_support=MIN_SUPPORT, use_colnames=True)
    regras = gerar_regras(itemsets, len(matriz), MIN_CONFIDENCE)
    regras = filtrar_lift_e_triviais(regras)
    curtas = filtrar_redundantes(regras, itemsets)
    sem_b = remover_modalidade_b(curtas)
    return uma_regra_por_itemset(sem_b)