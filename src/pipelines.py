"""Construção dos pipelines (pré-processamento + modelo) controlados pelo Optuna.

Cada pipeline tem até quatro passos, nesta ordem:
    imputador -> normalizador -> reducao -> modelo
O Optuna escolhe se cada passo opcional entra e com quais hiperparâmetros.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ALGORITMOS = ["knn", "arvore", "floresta", "gradient_boosting", "svm"]
ALGORITMOS_SENSIVEIS_A_ESCALA = ["knn", "svm"]

OPCOES_NORMALIZACAO = ["padrao", "robusto", "potencia"]
OPCOES_REDUCAO = ["nenhuma", "pca", "vif", "rfe"]


class SelecionadorVIF(BaseEstimator, TransformerMixin):
    """Remove atributos redundantes até todos terem VIF abaixo de um limiar.

    O VIF do atributo j é o j-ésimo elemento da diagonal da inversa da matriz
    de correlação, VIF_j = (R^-1)_jj = 1 / (1 - R²_j), onde R²_j mede o quanto
    os outros atributos explicam o atributo j. A cada passo, remove o atributo
    com maior VIF, até que todos fiquem abaixo do limiar.

    Args:
      limiar: VIF máximo aceito para os atributos mantidos.
    """

    def __init__(self, limiar=10.0):
        self.limiar = limiar

    def fit(self, X, y=None):
        """Escolhe as colunas mantidas usando só os dados recebidos (treino da dobra).

        Args:
          X: matriz de atributos (já normalizada).
          y: ignorado; existe só para seguir a interface do scikit-learn.

        Returns:
          O próprio seletor, com o atributo colunas_mantidas_.
        """
        X = np.asarray(X, dtype=float)

        correlacao = np.nan_to_num(np.corrcoef(X, rowvar=False))
        np.fill_diagonal(correlacao, 1.0)

        mantidos = list(range(X.shape[1]))

        while len(mantidos) > 1:
            sub_matriz = correlacao[np.ix_(mantidos, mantidos)]
            sub_matriz = sub_matriz + 1e-8 * np.eye(len(mantidos))  # evita matriz singular
            vif = np.diag(np.linalg.inv(sub_matriz))

            pior = int(np.argmax(vif))
            if vif[pior] <= self.limiar:
                break

            mantidos.pop(pior)

        self.colunas_mantidas_ = np.array(mantidos)
        return self

    def transform(self, X):
        """Mantém só as colunas escolhidas no fit.

        Args:
          X: matriz de atributos.

        Returns:
          Matriz só com as colunas mantidas.
        """
        return np.asarray(X, dtype=float)[:, self.colunas_mantidas_]


def cria_imputador(trial):
    """Imputação pela mediana, com indicador de faltante opcional.

    Args:
      trial: trial do Optuna (ou FixedTrial/FrozenTrial).

    Returns:
      Um SimpleImputer.
    """
    adiciona_indicador = trial.suggest_categorical("adiciona_indicador_faltante", [True, False])
    return SimpleImputer(strategy="median", add_indicator=adiciona_indicador)


def cria_normalizador(trial):
    """Escolhe entre normalização padrão, robusta ou de potência (Yeo-Johnson).

    Args:
      trial: trial do Optuna.

    Returns:
      Um StandardScaler, RobustScaler ou PowerTransformer.
    """
    escolha = trial.suggest_categorical("normalizacao", OPCOES_NORMALIZACAO)
    if escolha == "padrao":
        return StandardScaler()
    if escolha == "robusto":
        return RobustScaler()
    return PowerTransformer(method="yeo-johnson")


def cria_reducao(trial):
    """Escolhe a redução de dimensionalidade ou seleção de atributos.

    Args:
      trial: trial do Optuna.

    Returns:
      Um PCA, SelecionadorVIF, RFE ou None (sem redução).
    """
    metodo = trial.suggest_categorical("reducao", OPCOES_REDUCAO)

    if metodo == "pca":
        variancia = trial.suggest_float("pca_variancia_explicada", 0.80, 0.999)
        return PCA(n_components=variancia, svd_solver="full")

    if metodo == "vif":
        limiar = trial.suggest_float("vif_limiar", 2.0, 100.0, log=True)
        return SelecionadorVIF(limiar=limiar)

    if metodo == "rfe":
        fracao = trial.suggest_float("rfe_fracao_mantida", 0.2, 0.9)
        return RFE(
            estimator=LogisticRegression(max_iter=1000),
            n_features_to_select=fracao,
            step=0.1,
        )

    return None


def cria_modelo_knn(trial, semente_aleatoria):
    """k vizinhos mais próximos (a semente não é usada: o kNN é determinístico)."""
    return KNeighborsClassifier(
        n_neighbors=trial.suggest_int("knn_num_vizinhos", 1, 50),
        weights=trial.suggest_categorical("knn_pesos", ["uniform", "distance"]),
        p=trial.suggest_int("knn_p", 1, 2),
    )


def cria_modelo_arvore(trial, semente_aleatoria):
    """Árvore de decisão com poda por custo-complexidade."""
    return DecisionTreeClassifier(
        criterion=trial.suggest_categorical("arvore_criterio", ["gini", "entropy"]),
        max_depth=trial.suggest_int("arvore_profundidade_maxima", 2, 30),
        min_samples_leaf=trial.suggest_int("arvore_min_exemplos_folha", 1, 50, log=True),
        ccp_alpha=trial.suggest_float("arvore_ccp_alpha", 1e-5, 1e-1, log=True),
        random_state=semente_aleatoria,
    )


def cria_modelo_floresta(trial, semente_aleatoria):
    """Floresta aleatória (bagging de árvores com sorteio de atributos)."""
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("floresta_num_arvores", 100, 300),
        max_depth=trial.suggest_int("floresta_profundidade_maxima", 3, 30),
        min_samples_leaf=trial.suggest_int("floresta_min_exemplos_folha", 1, 20),
        max_features=trial.suggest_float("floresta_fracao_atributos", 0.1, 1.0),
        n_jobs=-1,
        random_state=semente_aleatoria,
    )


def cria_modelo_gradient_boosting(trial, semente_aleatoria):
    """Gradient boosting baseado em histogramas (árvores em sequência)."""
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("gb_taxa_aprendizado", 0.01, 0.3, log=True),
        max_iter=trial.suggest_int("gb_num_iteracoes", 100, 500),
        max_leaf_nodes=trial.suggest_int("gb_max_folhas", 8, 64),
        min_samples_leaf=trial.suggest_int("gb_min_exemplos_folha", 10, 100),
        l2_regularization=trial.suggest_float("gb_l2", 1e-6, 10.0, log=True),
        random_state=semente_aleatoria,
    )


def cria_modelo_svm(trial, semente_aleatoria):
    """Máquina de vetores de suporte com kernel RBF."""
    return SVC(
        kernel="rbf",
        C=trial.suggest_float("svm_c", 1e-2, 1e3, log=True),
        gamma=trial.suggest_float("svm_gamma", 1e-4, 1e1, log=True),
    )


CRIA_MODELO = {
    "knn": cria_modelo_knn,
    "arvore": cria_modelo_arvore,
    "floresta": cria_modelo_floresta,
    "gradient_boosting": cria_modelo_gradient_boosting,
    "svm": cria_modelo_svm,
}


def cria_instancia_modelo(trial, nome_algoritmo, semente_aleatoria):
    """Monta o pipeline completo: imputação, normalização, redução e modelo.

    A normalização só entra quando é necessária: nos modelos sensíveis à escala
    (kNN e SVM) ou quando há redução de atributos (PCA, VIF e RFE dependem da escala).

    Args:
      trial: trial do Optuna (Trial, FixedTrial ou FrozenTrial).
      nome_algoritmo: um dos nomes de ALGORITMOS.
      semente_aleatoria: semente dos modelos que têm aleatoriedade.

    Returns:
      Pipeline com os passos "imputador", "normalizador", "reducao" e "modelo"
      (os dois do meio só quando necessários).
    """
    imputador = cria_imputador(trial)
    reducao = cria_reducao(trial)

    precisa_normalizar = nome_algoritmo in ALGORITMOS_SENSIVEIS_A_ESCALA or reducao is not None
    normalizador = cria_normalizador(trial) if precisa_normalizar else None

    modelo = CRIA_MODELO[nome_algoritmo](trial, semente_aleatoria)

    passos = [("imputador", imputador)]
    if normalizador is not None:
        passos.append(("normalizador", normalizador))
    if reducao is not None:
        passos.append(("reducao", reducao))
    passos.append(("modelo", modelo))

    return Pipeline(passos)


def funcao_objetivo(trial, nome_algoritmo, X, y, grupos, cv, metrica, semente_aleatoria):
    """Função objetivo do Optuna: média da métrica na validação cruzada.

    Args:
      trial: trial do Optuna.
      nome_algoritmo: um dos nomes de ALGORITMOS.
      X: atributos de treino.
      y: target de treino.
      grupos: estrela (kepid) de cada exemplo, para a CV agrupada.
      cv: objeto de validação cruzada (sempre o mesmo para todos os modelos).
      metrica: nome da métrica do scikit-learn.
      semente_aleatoria: semente dos modelos.

    Returns:
      Média da métrica nas dobras.
    """
    pipeline = cria_instancia_modelo(trial, nome_algoritmo, semente_aleatoria)

    notas = cross_val_score(pipeline, X, y, groups=grupos, cv=cv, scoring=metrica)

    return notas.mean()
