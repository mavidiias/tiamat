"""Leitura dos dados do KOI e definição de atributos, target e grupos."""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

IDENTIFICADORES = ["kepid", "kepoi_name", "kepler_name", "ra_str", "dec_str"]
POSICAO_E_TEMPO = ["ra", "dec", "ra_err", "dec_err", "koi_time0bk", "koi_time0bk_err1",
                   "koi_time0bk_err2", "koi_time0", "koi_time0_err1", "koi_time0_err2"]
VAZAMENTO = ["koi_disposition", "koi_pdisposition", "koi_score", "koi_fpflag_nt",
             "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec", "koi_comment",
             "koi_disp_prov", "koi_vet_stat"]
METADADOS = ["koi_vet_date", "koi_delivname", "koi_tce_delivname", "koi_tce_plnt_num",
             "koi_quarters", "koi_trans_mod", "koi_limbdark_mod", "koi_fittype",
             "koi_parm_prov", "koi_sparprov", "koi_datalink_dvr", "koi_datalink_dvs"]
REMOVER = IDENTIFICADORES + POSICAO_E_TEMPO + VAZAMENTO + METADADOS


def carregar_dados(caminho="data/raw/koi_cumulative.csv"):
    """Lê a tabela cumulativa do KOI.

    Args:
      caminho: caminho do CSV bruto.

    Returns:
      DataFrame com todas as linhas e colunas.
    """
    return pd.read_csv(caminho)


def definir_colunas(df_treino):
    """Escolhe os atributos: tira as colunas de REMOVER e as constantes ou vazias.

    Args:
      df_treino: DataFrame de treino (as colunas constantes são vistas só no treino).

    Returns:
      Lista com os nomes dos atributos.
    """
    candidatas = [c for c in df_treino.columns if c not in REMOVER]
    return [c for c in candidatas if df_treino[c].nunique(dropna=True) > 1]


def preparar_X_y(df, colunas):
    """Separa atributos, target binário (1 = CONFIRMED) e grupos (estrela).

    Args:
      df: DataFrame de treino ou de teste.
      colunas: lista de atributos.

    Returns:
      Tupla (X, y, grupos).
    """
    X = df[colunas]
    y = (df["koi_disposition"] == "CONFIRMED").astype(int)
    grupos = df["kepid"]
    return X, y, grupos


def criar_cv(semente=810, n_splits=5):
    """Validação cruzada estratificada e agrupada por estrela.

    Args:
      semente: semente do embaralhamento das dobras.
      n_splits: número de dobras.

    Returns:
      Um StratifiedGroupKFold.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semente)
