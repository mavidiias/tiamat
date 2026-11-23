"""Otimização de hiperparâmetros com o Optuna para um algoritmo do projeto Tiamat.

Uso (sempre a partir da raiz do repositório):
    python -m src.optimize --algoritmo knn --trials 100
"""

import argparse
import json
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.exceptions import ConvergenceWarning

from src.data import criar_cv, definir_colunas, preparar_X_y
from src.pipelines import ALGORITMOS, funcao_objetivo

SEMENTE_ALEATORIA = 810
NUM_FOLDS_CV = 5
NUM_TRIALS = 100
METRICA_PRINCIPAL = "balanced_accuracy"
CAMINHO_TREINO = Path("data/processed/train.csv")
PASTA_RESULTADOS = Path("results/optuna")

# Hiperparâmetros padrão do scikit-learn (ou o valor mais próximo dentro da faixa
# de busca), enfileirados como primeiro trial de cada estudo.
HIPERPARAMETROS_COMUNS = {
    "adiciona_indicador_faltante": False,
    "reducao": "nenhuma",
}
HIPERPARAMETROS_PADRAO = {
    "knn": {
        "normalizacao": "padrao",
        "knn_num_vizinhos": 5,
        "knn_pesos": "uniform",
        "knn_p": 2,
    },
    "arvore": {
        "arvore_criterio": "gini",
        "arvore_profundidade_maxima": 30,      # padrão: None (sem limite)
        "arvore_min_exemplos_folha": 1,
        "arvore_ccp_alpha": 1e-5,              # padrão: 0 (fora da escala log)
    },
    "floresta": {
        "floresta_num_arvores": 100,
        "floresta_profundidade_maxima": 30,    # padrão: None (sem limite)
        "floresta_min_exemplos_folha": 1,
        "floresta_fracao_atributos": 0.11,     # padrão: sqrt(89) ≈ 9 atributos
    },
    "gradient_boosting": {
        "gb_taxa_aprendizado": 0.1,
        "gb_num_iteracoes": 100,
        "gb_max_folhas": 31,
        "gb_min_exemplos_folha": 20,
        "gb_l2": 1e-6,                         # padrão: 0 (fora da escala log)
    },
    "svm": {
        "normalizacao": "padrao",
        "svm_c": 1.0,
        "svm_gamma": 1 / 89,                   # padrão "scale" após StandardScaler
    },
}


def le_argumentos():
    """Lê os argumentos da linha de comando.

    Returns:
      Objeto com os atributos algoritmo, trials, semente, metrica, pasta_saida.
    """
    parser = argparse.ArgumentParser(description="Otimiza um algoritmo com o Optuna.")
    parser.add_argument("--algoritmo", required=True, choices=ALGORITMOS)
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--semente", type=int, default=SEMENTE_ALEATORIA)
    parser.add_argument("--metrica", default=METRICA_PRINCIPAL)
    parser.add_argument("--pasta-saida", type=Path, default=PASTA_RESULTADOS)
    return parser.parse_args()


def carrega_treino(caminho):
    """Lê o conjunto de treino e separa atributos, alvo e grupos.

    Args:
      caminho: caminho do train.csv.

    Returns:
      Tupla (X, y, grupos).
    """
    df_treino = pd.read_csv(caminho)
    atributos = definir_colunas(df_treino)
    return preparar_X_y(df_treino, atributos)


def cria_estudo(nome_algoritmo, pasta_saida, semente_aleatoria):
    """Cria (ou retoma) o estudo do Optuna salvo em SQLite.

    Args:
      nome_algoritmo: um dos nomes de ALGORITMOS.
      pasta_saida: pasta onde fica o arquivo <algoritmo>.db.
      semente_aleatoria: semente do amostrador TPE.

    Returns:
      O objeto optuna.Study.
    """
    pasta_saida.mkdir(parents=True, exist_ok=True)
    armazenamento = f"sqlite:///{pasta_saida / f'{nome_algoritmo}.db'}"

    amostrador = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        seed=semente_aleatoria,
    )

    estudo = optuna.create_study(
        study_name=nome_algoritmo,
        storage=armazenamento,
        sampler=amostrador,
        direction="maximize",
        load_if_exists=True,
    )

    padrao = HIPERPARAMETROS_COMUNS | HIPERPARAMETROS_PADRAO[nome_algoritmo]
    estudo.enqueue_trial(padrao, skip_if_exists=True)

    return estudo


def conta_trials_completos(estudo):
    """Conta os trials que terminaram com sucesso.

    Args:
      estudo: o objeto optuna.Study.

    Returns:
      Número de trials no estado COMPLETE.
    """
    return len(estudo.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))


def otimiza_ate_completar(estudo, objetivo, num_trials):
    """Roda o Optuna até haver num_trials trials completos.

    Trials que falham não contam; o laço roda de novo até completar.

    Args:
      estudo: o objeto optuna.Study.
      objetivo: função que recebe um trial e devolve a nota.
      num_trials: número de trials completos desejado.

    Returns:
      Número final de trials completos.
    """
    completos = conta_trials_completos(estudo)

    while completos < num_trials:
        estudo.optimize(
            objetivo,
            n_trials=num_trials - completos,
            catch=(ValueError,),
            show_progress_bar=True,
        )
        completos = conta_trials_completos(estudo)

    return completos


def salva_resultados(estudo, nome_algoritmo, metrica, semente_aleatoria, pasta_saida):
    """Salva a tabela de trials (CSV) e o melhor trial (JSON).

    Args:
      estudo: o objeto optuna.Study.
      nome_algoritmo: nome do algoritmo.
      metrica: nome da métrica otimizada.
      semente_aleatoria: semente usada.
      pasta_saida: pasta de saída.

    Returns:
      Dicionário salvo no JSON.
    """
    estudo.trials_dataframe().to_csv(pasta_saida / f"{nome_algoritmo}_trials.csv", index=False)

    melhores = {
        "algoritmo": nome_algoritmo,
        "metrica": metrica,
        "melhor_nota_cv": estudo.best_value,
        "numero_melhor_trial": estudo.best_trial.number,
        "melhores_hiperparametros": estudo.best_params,
        "trials_completos": conta_trials_completos(estudo),
        "semente_aleatoria": semente_aleatoria,
        "data": datetime.now().isoformat(timespec="seconds"),
    }

    with open(pasta_saida / f"{nome_algoritmo}_melhores.json", "w", encoding="utf-8") as arquivo:
        json.dump(melhores, arquivo, indent=2, ensure_ascii=False)

    return melhores


def main():
    """Lê os argumentos, roda a otimização e salva os resultados."""
    argumentos = le_argumentos()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # esconde a linha de cada trial

    X, y, grupos = carrega_treino(CAMINHO_TREINO)
    cv = criar_cv(argumentos.semente, NUM_FOLDS_CV)

    objetivo = partial(
        funcao_objetivo,
        nome_algoritmo=argumentos.algoritmo,
        X=X,
        y=y,
        grupos=grupos,
        cv=cv,
        metrica=argumentos.metrica,
        semente_aleatoria=argumentos.semente,
    )

    estudo = cria_estudo(argumentos.algoritmo, argumentos.pasta_saida, argumentos.semente)
    completos = otimiza_ate_completar(estudo, objetivo, argumentos.trials)
    melhores = salva_resultados(
        estudo, argumentos.algoritmo, argumentos.metrica, argumentos.semente, argumentos.pasta_saida
    )

    print(f"{argumentos.algoritmo}: {completos} trials completos")
    print(f"Melhor nota ({argumentos.metrica}): {melhores['melhor_nota_cv']:.4f}")
    print(f"Melhores hiperparâmetros: {melhores['melhores_hiperparametros']}")


if __name__ == "__main__":
    main()