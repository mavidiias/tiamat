# Tiamat: planeta ou falso positivo? Classificação de objetos de interesse do Kepler

**Autora:** Maria Vitória Dias Meneses
**Instituição:** Ilum Escola de Ciência, Centro Nacional de Pesquisa em Energia e Materiais (CNPEM)
**Disciplina:** Aprendizado de Máquina (ATP-203), professor Daniel Roberto Cassar
**Trabalho:** projeto final "4. Criatura lendária: Tiamat"

---

## Sumário

1. [Sobre o trabalho](#1-sobre-o-trabalho)
2. [Principais resultados](#2-principais-resultados)
3. [Estrutura do repositório](#3-estrutura-do-repositório)
4. [Como reproduzir o trabalho](#4-como-reproduzir-o-trabalho)
5. [Refazer a otimização do zero (opcional)](#5-refazer-a-otimização-do-zero-opcional)
6. [Problemas comuns](#6-problemas-comuns)
7. [Dados e citação](#7-dados-e-citação)
8. [Uso de inteligência artificial](#8-uso-de-inteligência-artificial)

---

## 1. Sobre o trabalho

A missão Kepler, da NASA, monitorou o brilho de mais de 150 mil estrelas e encontrou milhares de sinais com aparência de trânsito planetário, os **KOIs** (*Kepler Objects of Interest*). Muitos desses sinais, porém, são **falsos positivos**, causados principalmente por binárias eclipsantes e pela luz de outras estrelas.

Este trabalho compara algoritmos de aprendizado de máquina na tarefa de separar **planetas confirmados** de **falsos positivos**, a partir de 89 atributos numéricos da tabela cumulativa de KOIs do NASA Exoplanet Archive (7.587 KOIs de 6.641 estrelas).

O que foi feito:

- divisão treino/teste (80%/20%) **estratificada e agrupada por estrela**, com verificação de vazamento;
- análise exploratória completa, feita apenas no conjunto de treino;
- **5 algoritmos** (kNN, árvore de decisão, floresta aleatória, gradient boosting e SVM) otimizados com o **Optuna**, com **100 trials cada**, em *pipelines* que incluem imputação, normalização e **redução de dimensionalidade ou seleção de atributos** (PCA, VIF ou RFE) escolhidas pela própria otimização;
- comparação com **validação cruzada agrupada de 5 dobras** e com três baselines, tendo a **acurácia balanceada** como métrica principal;
- avaliação do modelo escolhido **uma única vez** no conjunto de teste;
- **explicabilidade** do modelo final com importância por permutação (de atributos e de grupos de atributos correlacionados) e gráficos de dependência parcial e ICE.

O relatório científico completo é o notebook **`notebook_tiamat.ipynb`**.

## 2. Principais resultados

| Modelo | Acurácia balanceada (validação cruzada, treino) | Acurácia balanceada (teste) |
|---|---|---|
| **Gradient boosting (escolhido)** | **0,976 ± 0,003** | **0,979** (IC 95%: 0,971 a 0,986) |
| SVM | 0,972 ± 0,004 | — |
| Floresta aleatória | 0,970 ± 0,003 | — |
| kNN | 0,964 ± 0,003 | — |
| Regressão logística (baseline forte) | 0,949 ± 0,006 | 0,954 |
| Árvore de decisão | 0,947 ± 0,009 | — |

Só o modelo escolhido e o baseline forte foram avaliados no teste, como manda a boa prática. O modelo final se apoia principalmente no **deslocamento do centroide** da imagem durante o trânsito, que indica que o sinal vem de outra estrela, seguido do raio do planeta, da órbita e da intensidade do sinal.

## 3. Estrutura do repositório

```
tiamat/
├── README.md                     este arquivo
├── requirements.txt              bibliotecas e versões usadas
├── .gitignore                    arquivos que o git deve ignorar
├── notebook_tiamat.ipynb         o relatório científico (o trabalho em si)
│
├── data/
│   ├── raw/
│   │   └── koi_cumulative.csv    tabela original do NASA Exoplanet Archive
│   └── processed/
│       ├── train.csv             conjunto de treino (6.070 KOIs)
│       └── test.csv              conjunto de teste (1.517 KOIs)
│
├── src/                          código compartilhado entre o notebook e o script
│   ├── __init__.py               marca a pasta src como um pacote Python
│   ├── data.py                   leitura dos dados, escolha das colunas e dobras
│   ├── pipelines.py              pipelines e espaços de busca do Optuna
│   └── optimize.py               script de otimização (linha de comando)
│
└── results/
    ├── optuna/                   estudos do Optuna (um por algoritmo)
    │   ├── <algoritmo>.db              estudo completo, em SQLite
    │   ├── <algoritmo>_trials.csv      tabela com todos os trials
    │   └── <algoritmo>_melhores.json   melhor nota e melhores hiperparâmetros
    └── cache/                    resultados salvos das contas mais demoradas
```

Os algoritmos são identificados pelos nomes `knn`, `arvore`, `floresta`, `gradient_boosting` e `svm`.

**Por que existem arquivos de resultados no repositório?** A otimização completa leva horas. Com os estudos do Optuna (`results/optuna/`) e o cache (`results/cache/`) já salvos, o notebook pode ser executado do início ao fim em poucos minutos, mostrando exatamente os resultados do relatório. Quem quiser refazer tudo do zero pode seguir a seção 5.

**Por que `train.csv` e `test.csv` estão salvos?** A divisão dos dados é feita uma única vez e salva, para que o conjunto de teste seja sempre o mesmo, no notebook e no script de otimização.

## 4. Como reproduzir o trabalho

Este é o caminho recomendado: ele executa o notebook inteiro usando os resultados de otimização já salvos e leva **poucos minutos**.

### 4.1 Pré-requisitos

- **Python 3.11 ou mais recente** ([python.org](https://www.python.org/downloads/)). Para conferir a versão instalada, rode `python --version` no terminal.
- **Git** ([git-scm.com](https://git-scm.com/)).
- Um ambiente para abrir notebooks: **JupyterLab** (instalado no passo 4.4) ou o **VS Code** com as extensões *Python* e *Jupyter*.

> **Atenção:** todos os comandos abaixo devem ser executados em um terminal aberto **na pasta raiz do repositório** (a pasta `tiamat/`, onde está este README).

### 4.2 Baixar o repositório

```bash
git clone <URL-DO-REPOSITÓRIO>
cd tiamat
```

Também é possível baixar o ZIP pelo botão **Code → Download ZIP** do GitHub e descompactá-lo. Nesse caso, abra o terminal dentro da pasta descompactada.

### 4.3 Criar um ambiente virtual

Um ambiente virtual isola as bibliotecas deste trabalho das demais instaladas no computador.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux ou macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Quando o ambiente está ativo, o nome `(.venv)` aparece no início da linha do terminal. Se você fechar o terminal, ative o ambiente de novo antes de continuar (só a segunda linha).

### 4.4 Instalar as bibliotecas

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

As versões de **NumPy, pandas, scikit-learn e Optuna** estão fixadas no `requirements.txt` porque os resultados dependem delas (veja a seção 6). Para conferir se a instalação deu certo:

```bash
python -c "import sklearn, optuna, numpy, pandas; print(sklearn.__version__, optuna.__version__, numpy.__version__, pandas.__version__)"
```

A saída esperada é `1.9.0 5.0.0 2.4.6 3.0.5`.

### 4.5 Abrir e executar o notebook

**Opção A, JupyterLab:**

```bash
jupyter lab
```

O navegador vai abrir. Clique em `notebook_tiamat.ipynb` e use o menu **Kernel → Restart Kernel and Run All Cells**.

**Opção B, VS Code:**

1. Abra a pasta `tiamat/` no VS Code (**File → Open Folder**).
2. Abra o arquivo `notebook_tiamat.ipynb`.
3. No canto superior direito, clique em **Select Kernel** e escolha o Python do ambiente `.venv`.
4. Clique em **Restart** e depois em **Run All**.

> **Importante:** o notebook precisa ser aberto **a partir da pasta raiz do repositório**, porque ele lê os arquivos com caminhos relativos (`data/...`, `results/...`) e importa os módulos de `src/`.

### 4.6 O que esperar

- O notebook roda de cima para baixo sem erros e leva **alguns minutos**. A etapa mais demorada é a seção 10 (explicabilidade), com cerca de 2 minutos.
- As contas lentas (estudo preliminar, baselines, reavaliação dos modelos e importâncias por permutação) são lidas de `results/cache/`, e os estudos do Optuna são lidos de `results/optuna/`. **Nenhuma otimização é refeita.**
- Com as versões indicadas, os números devem coincidir com os do relatório.

## 5. Refazer a otimização do zero (opcional)

Siga esta seção apenas se quiser **refazer os 500 trials** do Optuna (100 por algoritmo). Isso pode levar **várias horas**, dependendo do computador.

### 5.1 Apagar os resultados salvos

O script **retoma** um estudo existente: se o arquivo `.db` já tiver 100 trials completos, ele não faz nada. Para refazer, apague os resultados antigos.

**Windows (PowerShell):**

```powershell
Remove-Item results\optuna\* , results\cache\*
```

**Linux ou macOS:**

```bash
rm -f results/optuna/* results/cache/*
```

> Feche o notebook (ou reinicie o kernel) antes de apagar. Se o notebook estiver com um estudo aberto, o Windows não deixa apagar o `.db`.

### 5.2 Rodar a otimização de cada algoritmo

Sempre a partir da raiz do repositório, use `python -m src.optimize` (e não `python src/optimize.py`, que gera erro de importação):

```bash
python -m src.optimize --algoritmo knn --trials 100
python -m src.optimize --algoritmo arvore --trials 100
python -m src.optimize --algoritmo floresta --trials 100
python -m src.optimize --algoritmo gradient_boosting --trials 100
python -m src.optimize --algoritmo svm --trials 100
```

Cada comando mostra uma barra de progresso e, ao final, imprime a melhor nota e os melhores hiperparâmetros. Ele salva três arquivos em `results/optuna/`: o estudo (`<algoritmo>.db`), a tabela de trials (`<algoritmo>_trials.csv`) e o melhor trial (`<algoritmo>_melhores.json`).

Argumentos disponíveis:

| Argumento | Padrão | Descrição |
|---|---|---|
| `--algoritmo` | obrigatório | `knn`, `arvore`, `floresta`, `gradient_boosting` ou `svm` |
| `--trials` | `100` | número de trials **completos** desejado |
| `--semente` | `810` | semente do amostrador e das dobras |
| `--metrica` | `balanced_accuracy` | métrica otimizada (nome do scikit-learn) |
| `--pasta-saida` | `results/optuna` | pasta onde os resultados são salvos |

Dicas práticas:

- **Os algoritmos são independentes**, então podem ser rodados em terminais diferentes ao mesmo tempo, ou em computadores diferentes. Neste trabalho, a otimização foi dividida entre dois computadores com as mesmas versões das bibliotecas. Depois, basta copiar os arquivos de `results/optuna/` para o mesmo repositório.
- **Se a execução for interrompida**, rode o mesmo comando de novo: o estudo continua de onde parou.
- Na experiência deste trabalho, o kNN e a árvore levaram mais de uma hora cada em um computador pessoal, e a floresta foi a mais lenta por trial.

### 5.3 Executar o notebook

Depois que os cinco estudos terminarem, execute o notebook como na seção 4.5. Como o cache foi apagado, as contas lentas serão refeitas e salvas de novo em `results/cache/`. Isso acrescenta alguns minutos à execução; o Teste B do estudo preliminar é a etapa mais demorada.

### 5.4 Refazer também a divisão treino/teste (normalmente não é necessário)

A divisão é feita uma única vez, com a semente 810. O notebook só a refaz se os arquivos `data/processed/train.csv` e `data/processed/test.csv` **não existirem**. Para refazê-la, apague os dois arquivos **antes** de tudo e execute o notebook até a seção 3. Só então rode a otimização. Com a mesma semente e as mesmas versões, a divisão é idêntica.

## 6. Problemas comuns

| Problema | Causa provável | Solução |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | O notebook ou o script não foi aberto na raiz do repositório | Abra a pasta `tiamat/` e rode os comandos a partir dela; use `python -m src.optimize` |
| `FileNotFoundError` para `data/...` ou `results/...` | Mesma causa: pasta de trabalho errada | Mesma solução |
| `Activate.ps1 não pode ser carregado` no PowerShell | Política de execução de scripts do Windows | Rode `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e ative o ambiente de novo |
| Os números ficaram um pouco diferentes do relatório | Versões diferentes das bibliotecas | Instale exatamente as versões do `requirements.txt` |
| Mesmo com as versões certas, a nota do gradient boosting muda na terceira casa decimal | A seleção por VIF é numericamente sensível (discutido nas seções 8.2 e 11.5 do notebook) | Comportamento esperado em outro computador; o ranking dos modelos não muda |
| Erro ao apagar `results/optuna/*.db` no Windows | O arquivo está aberto pelo kernel do notebook | Reinicie ou feche o notebook e tente de novo |
| O notebook mostra resultados antigos depois de alterar o código | O cache em `results/cache/` foi reaproveitado | Apague o CSV correspondente ou mude `RECALCULAR = False` para `True` no notebook |

## 7. Dados e citação

Os dados são a **tabela cumulativa de KOIs do NASA Exoplanet Archive**, obtida pela API do arquivo em formato CSV e salva em `data/raw/koi_cumulative.csv`:

> NASA EXOPLANET ARCHIVE. **Kepler Objects of Interest Cumulative Table**. Pasadena: NExScI-Caltech/IPAC. DOI: [10.26133/NEA4](https://doi.org/10.26133/NEA4). Disponível em: https://exoplanetarchive.ipac.caltech.edu/.

A descrição de cada coluna está em [Data columns in the Kepler Objects of Interest table](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html). Como a tabela é atualizada periodicamente, o arquivo usado neste trabalho é mantido no repositório. A data de acesso e as demais referências estão na seção 13 do notebook.

Este trabalho usa os dados com o reconhecimento pedido pelo arquivo: *"This research has made use of the NASA Exoplanet Archive, which is operated by the California Institute of Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet Exploration Program."*

## 8. Uso de inteligência artificial

O trabalho foi desenvolvido com o apoio do assistente de IA Claude, da Anthropic. A descrição completa de como a IA foi usada nos códigos, nos textos e nas verificações está na seção **12. Descrição do uso de IA neste trabalho** do notebook.
