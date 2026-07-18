"""
================================================================
 MONITOR DE REPUTAÇÃO DE MARCA — APP de demonstração (Streamlit)
================================================================
APP de estudo/demonstração que simula um painel de monitoramento
de reputação de marca em redes sociais. Os dados usados são um
dataset público de tweets (Sentiment140) já pré-processados e
rotulados com sentimento (Positivo / Negativo / Neutro), servindo
apenas como EXEMPLO para apresentar o funcionamento do APP.

Como os tweets originais são de 2009 (sem timestamp útil para uma
demo de "monitoramento recente"), o APP gera datas SIMULADAS dos
últimos 30 dias apenas para ilustrar gráficos de tendência — isso
fica sinalizado na interface.

Para rodar:
    pip install -r requirements.txt
    streamlit run app.py
================================================================
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------------------------------------------
# CONFIGURAÇÃO GERAL
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Monitor de Reputação de Marca",
    page_icon="📢",
    layout="wide",
)

DATASETS = {
    "🇺🇸 Inglês — Sentiment140 (dados reais)": {
        "path": "tweets_com_sentimentos.csv",
        "idioma": "en",
        "aviso": (
            "Dataset público real de tweets (Sentiment140), em inglês, usado como exemplo."
        ),
    },
    "🇧🇷 Português — exemplo sintético": {
        "path": "tweets_com_sentimentos_pt.csv",
        "idioma": "pt",
        "aviso": (
            "Dataset **sintético** (gerado por templates, não são menções reais), só para "
            "demonstrar o APP funcionando em português. Para usar dados reais em PT, troque "
            "o arquivo por um dataset real com as mesmas colunas."
        ),
    },
}
SAMPLE_SIZE = 20_000
SEED = 42

COLORS = {"Positivo": "#2ecc71", "Negativo": "#e74c3c", "Neutro": "#95a5a6"}
CMAPS = {"Positivo": "Greens", "Negativo": "Reds", "Neutro": "Blues"}

LLM_MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
LLM_LABEL_MAP = {"positive": "Positivo", "negative": "Negativo", "neutral": "Neutro"}

STOPWORDS_PT = set("""
a à às ao aos as com como da das de dela dele deles delas do dos e é essa
essas esse esses esta estas este estes eu foi for foram fosse isso isto já
lhe lhes mais mas me mesmo meu meus minha minhas muito na nas nem no nos
nossa nossas nosso nossos num numa não o os ou para pela pelas pelo pelos
por qual quando que quem se seu seus sua suas só também te tem tinha
tive tu tua tuas tém têm um uma umas uns você vocês
""".split())


def limpar_texto(texto, idioma="en"):
    """Pipeline de limpeza de texto. Em inglês remove acentuação/caracteres
    não-ASCII e usa uma lista simples de stopwords em inglês; em português
    mantém acentos (importantes pro idioma) e usa stopwords em português."""
    import re
    import string as pystring

    t = texto.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    if idioma == "pt":
        t = t.translate(str.maketrans("", "", pystring.punctuation))
        t = re.sub(r"\d+", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        tokens = [w for w in t.split() if w not in STOPWORDS_PT and len(w) > 2]
    else:
        t = re.sub(r"[^\x00-\x7F]+", " ", t)
        t = t.translate(str.maketrans("", "", pystring.punctuation))
        t = re.sub(r"\d+", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        stop_en = {"the", "and", "for", "you", "your", "with", "this", "that", "was", "are"}
        tokens = [w for w in t.split() if w not in stop_en and len(w) > 2]
    return " ".join(tokens)


# ----------------------------------------------------------------
# CARGA E PREPARO DOS DADOS (dados de exemplo)
# ----------------------------------------------------------------
REQUIRED_COLUMNS = {"clean_text", "sentiment", "compound", "pos", "neg", "neu"}


def _arquivo_assinatura(path):
    """Retorna (mtime, tamanho) do arquivo. Usado como parte da chave de cache,
    pra detectar automaticamente quando o CSV foi substituído por outro
    (mesmo nome, conteúdo diferente) sem precisar reiniciar o APP."""
    try:
        stat = os.stat(path)
        return (stat.st_mtime, stat.st_size)
    except OSError:
        return (0, 0)


def _read_csv_robusto(path):
    """Lê o CSV tentando detectar automaticamente o separador (',' ou ';').

    Um erro comum é o arquivo ter sido aberto e salvo pelo Excel em
    configuração PT-BR, que troca a vírgula por ponto-e-vírgula como
    separador — nesse caso o pandas, com separador fixo, leria tudo como
    uma única coluna. Aqui tentamos primeiro com detecção automática
    (sep=None) e, se falhar, caímos para vírgula e ponto-e-vírgula.
    """
    tentativas = [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
    ]
    ultimo_erro = None
    for kwargs in tentativas:
        try:
            df = pd.read_csv(path, **kwargs)
        except Exception as e:  # noqa: BLE001
            ultimo_erro = e
            continue
        if REQUIRED_COLUMNS.issubset(set(df.columns)):
            return df
    # Nenhuma tentativa produziu as colunas esperadas
    colunas_encontradas = list(df.columns) if "df" in dir() else "desconhecidas"
    raise ValueError(
        f"Não consegui encontrar as colunas esperadas ({sorted(REQUIRED_COLUMNS)}) "
        f"em '{path}'. Colunas encontradas: {colunas_encontradas}. "
        "Verifique se o arquivo não foi reaberto/resalvo pelo Excel (isso costuma "
        "trocar a vírgula por ponto-e-vírgula como separador) e se é o mesmo "
        "'tweets_com_sentimentos.csv' gerado originalmente."
    ) from ultimo_erro


@st.cache_data(show_spinner="Carregando dados de exemplo...")
def load_data(data_path, assinatura_arquivo, sample_size=SAMPLE_SIZE, seed=SEED):
    """`assinatura_arquivo` (mtime + tamanho) não é usado dentro da função — serve
    só pra forçar o Streamlit a recarregar automaticamente quando alguém troca o
    conteúdo do CSV, mesmo mantendo o mesmo nome/caminho de arquivo."""
    df = _read_csv_robusto(data_path)
    if "5" in df.columns:
        df = df.rename(columns={"5": "texto_original"})
    elif "texto_original" not in df.columns:
        # fallback: primeira coluna que não faz parte das colunas conhecidas
        candidatas = [c for c in df.columns if c not in REQUIRED_COLUMNS]
        df = df.rename(columns={candidatas[0]: "texto_original"}) if candidatas else df
    df["clean_text"] = df["clean_text"].fillna("")
    if "texto_original" in df.columns:
        df["texto_original"] = df["texto_original"].fillna("")
    else:
        df["texto_original"] = ""
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    if len(df) > sample_size:
        # Amostragem estratificada por sentimento SEM usar groupby().apply():
        # em versões recentes do pandas, .apply() sobre um groupby remove a
        # própria coluna de agrupamento do resultado (era só um FutureWarning
        # antes, virou comportamento padrão) — então fazemos isso manualmente.
        frac = sample_size / len(df)
        partes = []
        for _, grupo in df.groupby("sentiment"):
            n = max(1, int(len(grupo) * frac))
            partes.append(grupo.sample(n, random_state=seed))
        df = pd.concat(partes, ignore_index=True)

    # Datas simuladas (o dataset original não tem timestamp útil p/ demo)
    rng = np.random.default_rng(seed)
    end = datetime.now().date()
    start = end - timedelta(days=29)
    span = (end - start).days
    offsets = rng.integers(0, span + 1, size=len(df))
    df["data"] = [start + timedelta(days=int(o)) for o in offsets]

    return df


@st.cache_data(show_spinner=False)
def make_wordcloud_image(text, colormap):
    if not text.strip():
        return None
    wc = WordCloud(
        width=900, height=500, background_color="white",
        colormap=colormap, max_words=120, collocations=False,
    ).generate(text)
    return wc.to_array()


@st.cache_resource(show_spinner=False)
def get_vader():
    return SentimentIntensityAnalyzer()


# ----------------------------------------------------------------
# ML CLÁSSICO (Regressão Logística + Naive Bayes sobre TF-IDF)
# ----------------------------------------------------------------
@st.cache_resource(show_spinner="Treinando modelos de ML clássico...")
def train_ml_models(_df, cache_key):
    """`_df` (com underscore) não entra no cálculo do cache do Streamlit — por
    isso passamos `cache_key` (idioma + assinatura do arquivo) só para garantir
    que o cache é invalidado de verdade quando o dataset muda: seja ao trocar
    o idioma na barra lateral, seja ao substituir o conteúdo do CSV no disco."""
    X = _df["clean_text"]
    y = _df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    candidatos = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Naive Bayes (Complement)": ComplementNB(),
    }

    resultados = {}
    for nome, clf in candidatos.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", clf),
        ])
        t0 = time.time()
        pipe.fit(X_train, y_train)
        tempo_treino = time.time() - t0

        t0 = time.time()
        y_pred = pipe.predict(X_test)
        tempo_infer = (time.time() - t0) / max(len(X_test), 1)

        resultados[nome] = {
            "pipeline": pipe,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "tempo_treino": tempo_treino,
            "tempo_infer_ms": tempo_infer * 1000,
            "y_test": y_test,
            "y_pred": y_pred,
            "report": classification_report(y_test, y_pred, output_dict=True),
        }
    return resultados


# ----------------------------------------------------------------
# DEEP LEARNING / LLM (BERTimbau-style multilingual distilbert)
# ----------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_llm_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=LLM_MODEL_NAME)


def run_llm_eval(df, n_amostra=200):
    clf = load_llm_pipeline()
    amostra = df.sample(n=min(n_amostra, len(df)), random_state=SEED)

    preds, tempos = [], []
    progress = st.progress(0.0, text="Classificando amostra com o modelo de Deep Learning...")
    for i, txt in enumerate(amostra["clean_text"]):
        t0 = time.time()
        out = clf(txt[:512] if txt.strip() else "neutro")[0]
        tempos.append(time.time() - t0)
        preds.append(LLM_LABEL_MAP.get(out["label"].lower(), out["label"]))
        progress.progress((i + 1) / len(amostra))
    progress.empty()

    y_true = amostra["sentiment"].values
    y_pred = np.array(preds)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "tempo_infer_ms": np.mean(tempos) * 1000,
        "y_test": y_true,
        "y_pred": y_pred,
        "n_amostra": len(amostra),
    }


# ----------------------------------------------------------------
# COMPONENTES DE VISUALIZAÇÃO
# ----------------------------------------------------------------
def plot_confusion(y_test, y_pred, labels, title, figsize=(4.2, 3.6)):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels,
                yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title, fontsize=11)
    st.pyplot(fig)
    plt.close(fig)


def sentiment_badge(label):
    color = COLORS.get(label, "#888")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:10px;font-size:0.85em;">{label}</span>'


def render_rodape(nome_dataset, info_dataset):
    """Rodapé fixo no final de cada página, mostrando qual dataset está em análise."""
    st.divider()
    st.markdown(f"📁 **Dataset em análise:** {nome_dataset}")
    st.caption(f"⚠️ Demonstração: {info_dataset['aviso']} As **datas são simuladas**.")


def aplicar_filtro_marca(df_base, marca, chave_widget):
    """Checkbox padrão de filtro por palavra-chave, usado em todas as páginas da
    seção 'Monitoramento da Marca'. Retorna o DataFrame filtrado (ou o completo,
    se o filtro estiver desligado ou não encontrar nada)."""
    ativo = st.checkbox(
        f"Filtrar apenas menções que contêm '{marca}'", value=True, key=chave_widget,
        help="Desmarque para ver com todo o dataset de exemplo, sem filtro.",
    )
    if ativo and marca.strip():
        filtrado = df_base[df_base["texto_original"].str.contains(marca, case=False, na=False)]
        if filtrado.empty:
            st.warning(
                f"Nenhuma menção encontrada contendo **'{marca}'**. Mostrando o dataset completo "
                "como alternativa — tente outra palavra-chave na barra lateral."
            )
            return df_base
        st.caption(f"Mostrando **{len(filtrado):,}** de {len(df_base):,} menções que contêm '{marca}'.")
        return filtrado
    return df_base


def interpretacao(texto):
    """Caixa padronizada de interpretação de resultados, logo abaixo de um gráfico."""
    st.markdown(f"📌 **Interpretação:** {texto}")


def interpretar_confusao(y_test, y_pred, labels, nome_modelo):
    """Gera um texto automático apontando a classe com melhor acerto e a maior confusão."""
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    acertos_pct = {lab: (cm[i, i] / cm[i].sum() * 100 if cm[i].sum() else 0) for i, lab in enumerate(labels)}
    melhor = max(acertos_pct, key=acertos_pct.get)
    pior = min(acertos_pct, key=acertos_pct.get)

    maior_confusao = None
    maior_valor = -1
    for i, real in enumerate(labels):
        for j, predito in enumerate(labels):
            if i != j and cm[i, j] > maior_valor:
                maior_valor = cm[i, j]
                maior_confusao = (real, predito)

    texto = (
        f"O **{nome_modelo}** acerta melhor a classe **{melhor}** ({acertos_pct[melhor]:.0f}% de acerto nela) "
        f"e tem mais dificuldade com **{pior}** ({acertos_pct[pior]:.0f}%). "
    )
    if maior_confusao and maior_valor > 0:
        texto += (
            f"O erro mais comum é confundir **{maior_confusao[0]}** com **{maior_confusao[1]}** "
            f"({maior_valor} casos na amostra de teste) — normalmente acontece com textos mais neutros/ambíguos."
        )
    interpretacao(texto)


# ----------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------
st.sidebar.title("📢 Monitor de Marca")

idioma_escolhido = st.sidebar.selectbox("🌐 Idioma dos dados", list(DATASETS.keys()))
dataset_info = DATASETS[idioma_escolhido]
DATA_PATH = dataset_info["path"]
IDIOMA = dataset_info["idioma"]  # "en" ou "pt"

PAGINAS_MARCA = [
    "🏠 Visão Geral",
    "🔍 Buscar Menções",
    "☁️ Nuvens de Palavras",
    "📊 Sentimentos & Score",
]
PAGINAS_LABORATORIO = [
    "🧪 Pré-processamento",
    "🤖 ML Clássico",
    "🧠 Deep Learning (LLM)",
    "⚖️ Comparação de Modelos",
]

secao = st.sidebar.radio(
    "Seção",
    ["📖 Início", "📌 Monitoramento da Marca", "🧪 Laboratório de IA"],
    help=(
        "Início: explica o que é sentimento Positivo/Negativo/Neutro e as técnicas usadas. "
        "Monitoramento da Marca: páginas que filtram por uma palavra-chave. "
        "Laboratório de IA: páginas que sempre usam o dataset completo e classificam a frase abaixo."
    ),
)

marca, frase = None, None
if secao == "📖 Início":
    pagina = "📖 Início"
elif secao == "📌 Monitoramento da Marca":
    marca = st.sidebar.text_input(
        "Nome da marca / palavra-chave a monitorar",
        value="marca" if IDIOMA == "pt" else "love",
        help="Usado no título do painel e como palavra-chave padrão na página 'Buscar Menções'.",
    )
    st.sidebar.caption("Estas páginas filtram pela palavra-chave acima.")
    pagina = st.sidebar.radio("Navegação", PAGINAS_MARCA)
else:
    frase = st.sidebar.text_area(
        "Frase para testar",
        value=("Adorei o atendimento dessa marca, muito rápido e profissional!" if IDIOMA == "pt"
               else "I really love this brand, amazing service!"),
        help="Usada em todas as páginas abaixo para mostrar como cada técnica classificaria essa frase.",
        height=80,
    )
    st.sidebar.caption("Estas páginas sempre usam o dataset completo pra métricas, e a frase acima pra classificar.")
    pagina = st.sidebar.radio("Navegação", PAGINAS_LABORATORIO)

try:
    df = load_data(DATA_PATH, _arquivo_assinatura(DATA_PATH))
except FileNotFoundError:
    st.error(
        f"Não encontrei o arquivo `{DATA_PATH}` na pasta do projeto.\n\n"
        + ("Gere-o com `python gerar_dataset_pt.py` (deixei esse script na pasta) e recarregue a página."
           if IDIOMA == "pt" else
           "Confirme se o arquivo está na mesma pasta do `app.py`.")
    )
    st.stop()

if "sentiment" not in df.columns:
    st.error(
        "A coluna **'sentiment'** não foi encontrada no DataFrame carregado.\n\n"
        f"Colunas disponíveis: `{list(df.columns)}`\n\n"
        f"Shape: {df.shape}\n\n"
        f"Caminho lido: `{DATA_PATH}`"
    )
    st.stop()

counts = df["sentiment"].value_counts()
labels_ordenados = ["Positivo", "Neutro", "Negativo"]

# ----------------------------------------------------------------
# PÁGINA: INÍCIO
# ----------------------------------------------------------------
if pagina == "📖 Início":
    st.markdown(
        "<h1 style='text-align: center;'>Este APP analisa o perfil e o sentimento dos seus clientes "
        "para apoiar decisões de compra e retenção</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("O APP é dividido em duas seções, escolhidas na barra lateral — cada uma responde a um tipo de pergunta diferente:")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📌 Por que existe o Monitoramento da Marca?")
        st.markdown(
            "Filtra o dataset de exemplo por uma palavra-chave (nome da marca, produto, campanha...), "
            "simulando o acompanhamento de menções reais nas redes sociais. Responde perguntas do dia "
            "a dia do negócio, como: *o que estão falando sobre nós? está tendo mais elogio ou "
            "reclamação essa semana? teve algum pico de crise?* Por isso todas as páginas dessa seção "
            "(Visão Geral, Buscar Menções, Nuvens de Palavras, Sentimentos & Score) filtram pela "
            "mesma palavra-chave digitada na barra lateral."
        )
    with col_b:
        st.subheader("🧪 Por que existe o Laboratório de IA?")
        st.markdown(
            "Aqui o foco é técnico: entender e comparar **como** a classificação de sentimento é "
            "feita por trás dos panos — da limpeza do texto até a decisão final de cada modelo "
            "(VADER, Naive Bayes, Regressão Logística e Deep Learning). Por isso essas páginas sempre "
            "usam o dataset completo pra medir desempenho (não faz sentido 'filtrar' a métrica de um "
            "modelo), e usam a frase digitada na barra lateral pra mostrar, na prática, o que cada "
            "técnica decide para o mesmo texto."
        )

    st.header("😀 O que significam Positivo, Negativo e Neutro?")
    st.markdown(
        "Cada menção (tweet, comentário, avaliação...) é classificada em uma dessas três categorias "
        "de acordo com o sentimento predominante no texto. Isso não é só um rótulo — é um dado que "
        "orienta decisões de negócio:"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"### {sentiment_badge('Positivo')}", unsafe_allow_html=True)
        st.markdown(
            "Elogios, satisfação, recomendação. \n\n**Uso em negócio:** identificar embaixadores da "
            "marca, casos de sucesso pra usar em depoimentos/marketing, e o que está funcionando bem "
            "e deve ser mantido."
        )
    with c2:
        st.markdown(f"### {sentiment_badge('Neutro')}", unsafe_allow_html=True)
        st.markdown(
            "Menções informativas, dúvidas, comentários sem carga emocional forte. \n\n**Uso em "
            "negócio:** mapear dúvidas frequentes (oportunidade de melhorar FAQ/comunicação), sem "
            "indicar urgência de resposta."
        )
    with c3:
        st.markdown(f"### {sentiment_badge('Negativo')}", unsafe_allow_html=True)
        st.markdown(
            "Reclamações, frustração, insatisfação. \n\n**Uso em negócio:** sinaliza risco de perda "
            "de clientes (churn) e de crise de reputação — normalmente é a categoria que pede "
            "resposta mais rápida do time de atendimento/PR."
        )
    st.markdown(
        "Na prática, acompanhar a **proporção** dessas três categorias ao longo do tempo (como no "
        "gráfico de tendência da Visão Geral) ajuda a detectar uma crise começando, antes que ela "
        "cresça — por isso o APP tem um alerta automático para dias com muitas menções negativas."
    )

    st.header("🧠 Quais técnicas o APP usa para classificar sentimento?")
    st.markdown(
        "O **Laboratório de IA** compara quatro abordagens diferentes, cada uma com vantagens e "
        "limitações. Não existe uma 'escolha certa' universal — depende do volume de dados, "
        "orçamento e necessidade de precisão:"
    )

    tecnicas = pd.DataFrame([
        {
            "Técnica": "VADER (léxico)",
            "Como funciona": "Dicionário de palavras com peso de sentimento pré-definido (ex.: 'ótimo' = +, 'péssimo' = -). Soma os pesos do texto.",
            "Velocidade": "Muito rápida",
            "Precisa treinar?": "Não",
            "Entende contexto/ironia?": "Não",
        },
        {
            "Técnica": "Naive Bayes",
            "Como funciona": "Modelo probabilístico (Teorema de Bayes) que estima a chance de cada classe a partir da frequência das palavras, assumindo que elas são independentes entre si.",
            "Velocidade": "Muito rápida",
            "Precisa treinar?": "Sim (rápido)",
            "Entende contexto/ironia?": "Pouco",
        },
        {
            "Técnica": "Regressão Logística",
            "Como funciona": "Modelo estatístico linear que aprende um peso por palavra (via TF-IDF) associado a cada classe. Bom equilíbrio entre simplicidade e desempenho.",
            "Velocidade": "Rápida",
            "Precisa treinar?": "Sim (rápido)",
            "Entende contexto/ironia?": "Pouco",
        },
        {
            "Técnica": "Deep Learning (Transformer/LLM)",
            "Como funciona": "Rede neural profunda (ex.: BERT/DistilBERT) pré-treinada em muito texto, que entende a ordem das palavras, negação e contexto antes de classificar.",
            "Velocidade": "Lenta (precisa de mais poder computacional)",
            "Precisa treinar?": "Não (já vem pronto, só ajustado)",
            "Entende contexto/ironia?": "Sim, bem melhor",
        },
    ])
    st.dataframe(tecnicas, use_container_width=True, hide_index=True)

    st.markdown(
        "**Na prática:** técnicas rápidas (VADER, Naive Bayes, Regressão Logística) são melhores "
        "para monitorar um volume alto de menções continuamente; o Deep Learning custa mais caro e "
        "é mais lento, mas vale a pena para casos mais ambíguos ou de alto risco, onde precisão "
        "importa mais que velocidade. Muitas empresas usam as duas: uma técnica rápida pra triagem "
        "geral, e o modelo mais pesado só nos casos que precisam de mais confiança."
    )

    st.info(
        "👉 Vá até **🧪 Laboratório de IA** na barra lateral para ver essas quatro técnicas "
        "classificando a mesma frase, lado a lado."
    )

# ----------------------------------------------------------------
# PÁGINA: VISÃO GERAL
# ----------------------------------------------------------------
elif pagina == "🏠 Visão Geral":
    st.title(f"📢 Monitor de Reputação — {marca}")
    st.caption("Painel de exemplo com dados públicos de demonstração.")

    df_geral = aplicar_filtro_marca(df, marca, "filtro_visao_geral")
    counts_geral = df_geral["sentiment"].value_counts()

    c1, c2, c3, c4 = st.columns(4)
    pct_pos = 100 * counts_geral.get("Positivo", 0) / len(df_geral)
    pct_neg = 100 * counts_geral.get("Negativo", 0) / len(df_geral)
    pct_neu = 100 * counts_geral.get("Neutro", 0) / len(df_geral)
    score_medio = df_geral["compound"].mean()
    c1.metric("Menções analisadas", f"{len(df_geral):,}")
    c2.metric("% Positivas", f"{pct_pos:.1f}%")
    c3.metric("% Negativas", f"{pct_neg:.1f}%")
    c4.metric("Score médio (compound)", f"{score_medio:.3f}")

    saldo = pct_pos - pct_neg
    if saldo > 5:
        leitura_saldo = f"saldo **positivo** de {saldo:.1f} pontos percentuais (mais elogios do que reclamações)"
    elif saldo < -5:
        leitura_saldo = f"saldo **negativo** de {abs(saldo):.1f} pontos percentuais (mais reclamações do que elogios)"
    else:
        leitura_saldo = "saldo praticamente **equilibrado** entre elogios e reclamações"
    interpretacao(
        f"De {len(df_geral):,} menções analisadas, {pct_pos:.1f}% são positivas, {pct_neg:.1f}% negativas e "
        f"{pct_neu:.1f}% neutras — {leitura_saldo}. O score médio de {score_medio:.3f} "
        f"({'acima' if score_medio > 0 else 'abaixo' if score_medio < 0 else 'exatamente'} de zero) reforça essa leitura."
    )

    st.subheader("Tendência de menções (últimos 30 dias — datas simuladas)")
    trend = (
        df_geral.groupby(["data", "sentiment"]).size().unstack(fill_value=0)
        .reindex(columns=labels_ordenados, fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.stackplot(trend.index, [trend[c] for c in labels_ordenados],
                 colors=[COLORS[c] for c in labels_ordenados], labels=labels_ordenados)
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylabel("Nº de menções")
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

    dia_pico = trend.sum(axis=1).idxmax()
    vol_pico = int(trend.sum(axis=1).max())
    dia_mais_neg = trend["Negativo"].idxmax()
    vol_mais_neg = int(trend["Negativo"].max())
    interpretacao(
        f"Como as datas são simuladas, a tendência fica praticamente estável ao longo dos 30 dias "
        f"(volume diário distribuído de forma aleatória). O dia com mais menções no total foi "
        f"**{dia_pico}** ({vol_pico} menções), e o dia com mais menções negativas foi **{dia_mais_neg}** "
        f"({vol_mais_neg}). Com dados reais, esse gráfico é o principal indicador para detectar picos "
        f"de crise de reputação ao longo do tempo."
    )

    st.subheader("⚠️ Alertas — dias com mais de 45% de menções negativas")
    daily_pct = trend.div(trend.sum(axis=1), axis=0) * 100
    alertas = daily_pct[daily_pct["Negativo"] > 45].sort_values("Negativo", ascending=False)
    if alertas.empty:
        st.success("Nenhum dia com pico relevante de menções negativas na amostra.")
        interpretacao(
            "Nenhum dia ultrapassou 45% de menções negativas nesta amostra simulada — em um cenário "
            "real, esse limite serviria de gatilho para investigar a causa (ex.: falha em produto, "
            "campanha mal recebida) antes que o problema cresça."
        )
    else:
        st.dataframe(
            alertas[["Negativo"]].rename(columns={"Negativo": "% Negativo"}).round(1),
            use_container_width=True,
        )
        pior_dia = alertas.index[0]
        interpretacao(
            f"{len(alertas)} dia(s) passaram do limite de 45% de menções negativas, com destaque para "
            f"**{pior_dia}** ({alertas.iloc[0]['Negativo']:.1f}%). Esses seriam os primeiros pontos a "
            f"investigar em um monitoramento real."
        )

    st.subheader("Menções recentes")
    recentes = df_geral.sort_values("data", ascending=False).head(8)[["data", "texto_original", "sentiment"]]
    for _, row in recentes.iterrows():
        st.markdown(
            f"**{row['data']}** — {sentiment_badge(row['sentiment'])}<br>{row['texto_original']}",
            unsafe_allow_html=True,
        )
        st.divider()

# ----------------------------------------------------------------
# PÁGINA: BUSCAR MENÇÕES
# ----------------------------------------------------------------
elif pagina == "🔍 Buscar Menções":
    st.title("🔍 Buscar Menções")
    st.caption(
        "Simule a busca por uma palavra-chave/marca dentro dos tweets de exemplo. "
        "Já vem preenchido com o que você digitou em 'Nome da marca' na barra lateral — "
        "pode trocar aqui sem afetar o valor da barra lateral."
    )

    termo = st.text_input("Palavra-chave", value=marca)
    if termo.strip():
        filtrado = df[df["texto_original"].str.contains(termo, case=False, na=False)]
        st.write(f"**{len(filtrado):,}** menções encontradas para `{termo}`.")

        if len(filtrado) > 0:
            sub_counts = filtrado["sentiment"].value_counts().reindex(labels_ordenados, fill_value=0)
            col1, col2 = st.columns([1, 1.4])
            with col1:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(sub_counts.values, labels=sub_counts.index,
                       colors=[COLORS[c] for c in sub_counts.index],
                       autopct="%1.0f%%", startangle=140)
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                st.dataframe(sub_counts.rename("Menções"), use_container_width=True)

            dominante = sub_counts.idxmax()
            pct_dominante = 100 * sub_counts.max() / sub_counts.sum()
            interpretacao(
                f"Entre as menções que contêm '{termo}', a maioria é **{dominante}** "
                f"({pct_dominante:.0f}% do total). Isso dá uma leitura rápida de como esse termo/assunto "
                f"está sendo percebido dentro da amostra."
            )

            texto_wc = " ".join(filtrado["clean_text"])
            img = make_wordcloud_image(texto_wc, "viridis")
            if img is not None:
                st.image(img, caption=f"Nuvem de palavras para '{termo}'", use_container_width=True)
                interpretacao(
                    "As palavras maiores são as que mais aparecem junto com o termo buscado — útil para "
                    "entender rapidamente o contexto em que a marca/palavra-chave é mencionada."
                )

            st.subheader("Exemplos de menções")
            amostra = filtrado.sample(min(10, len(filtrado)), random_state=SEED)
            for _, row in amostra.iterrows():
                st.markdown(
                    f"{sentiment_badge(row['sentiment'])} &nbsp; {row['texto_original']}",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Nenhuma menção encontrada com esse termo. Tente outra palavra.")

# ----------------------------------------------------------------
# PÁGINA: NUVENS DE PALAVRAS
# ----------------------------------------------------------------
elif pagina == "☁️ Nuvens de Palavras":
    st.title("☁️ Nuvens de Palavras por Sentimento")
    df_pg = aplicar_filtro_marca(df, marca, "filtro_nuvens")

    cols = st.columns(3)
    palavra_top_por_sentimento = {}
    for col, sent in zip(cols, labels_ordenados):
        texto = " ".join(df_pg[df_pg["sentiment"] == sent]["clean_text"])
        img = make_wordcloud_image(texto, CMAPS[sent])
        if texto.split():
            palavra_top_por_sentimento[sent] = pd.Series(texto.split()).value_counts().idxmax()
        with col:
            st.markdown(f"**{sent}** ({(df_pg['sentiment']==sent).sum():,} menções)")
            if img is not None:
                st.image(img, use_container_width=True)
            else:
                st.caption("Sem menções suficientes para gerar a nuvem.")

    palavras_repetidas = set.intersection(*[
        set(" ".join(df_pg[df_pg["sentiment"] == s]["clean_text"]).split()) for s in labels_ordenados
    ]) if len(labels_ordenados) > 1 else set()
    top_repetidas = sorted(palavras_repetidas)[:3] if palavras_repetidas else []
    if palavra_top_por_sentimento:
        interpretacao(
            "Quanto maior a palavra, mais vezes ela aparece nos textos daquele sentimento. Palavra mais "
            "frequente em cada grupo: " +
            ", ".join(f"**{s}** → *{w}*" for s, w in palavra_top_por_sentimento.items()) +
            (f". Palavras que aparecem nos três grupos (ex.: {', '.join(top_repetidas)}) tendem a ser termos "
             "genéricos do dia a dia, não indicadores fortes de sentimento — por isso o ideal é olhar também "
             "para as palavras exclusivas de cada nuvem." if top_repetidas else "")
        )

    st.subheader("Top 15 palavras por sentimento")
    cols2 = st.columns(3)
    for col, sent in zip(cols2, labels_ordenados):
        palavras = " ".join(df_pg[df_pg["sentiment"] == sent]["clean_text"]).split()
        if not palavras:
            continue
        top = pd.Series(palavras).value_counts().head(15).sort_values()
        fig, ax = plt.subplots(figsize=(4, 4.5))
        ax.barh(top.index, top.values, color=COLORS[sent])
        ax.set_title(sent, fontsize=11)
        with col:
            st.pyplot(fig)
            plt.close(fig)

    interpretacao(
        "Este gráfico mostra a contagem exata (a nuvem acima só dá uma noção visual do tamanho). Serve "
        "para confirmar, com números, se uma palavra realmente é relevante antes de tirar conclusões só "
        "olhando o tamanho dela na nuvem."
    )

# ----------------------------------------------------------------
# PÁGINA: SENTIMENTOS & SCORE
# ----------------------------------------------------------------
elif pagina == "📊 Sentimentos & Score":
    st.title("📊 Distribuição de Sentimentos")
    if IDIOMA == "pt":
        st.caption(
            "Neste dataset sintético em português, o 'compound score' é gerado junto com o texto "
            "(não vem do VADER, que só funciona bem em inglês) — serve só para ilustrar como esse "
            "tipo de gráfico funciona."
        )

    df_pg = aplicar_filtro_marca(df, marca, "filtro_score")
    counts_pg = df_pg["sentiment"].value_counts().reindex(labels_ordenados, fill_value=0)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(counts_pg.index, counts_pg.values, color=[COLORS[s] for s in counts_pg.index])
        for i, v in enumerate(counts_pg.values):
            ax.text(i, v + max(counts_pg.values, default=0) * 0.01, f"{v:,}", ha="center", fontweight="bold")
        ax.set_title("Contagem por sentimento")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts_pg.values, labels=counts_pg.index, colors=[COLORS[s] for s in counts_pg.index],
               autopct="%1.1f%%", startangle=140)
        ax.set_title("Proporção por sentimento")
        st.pyplot(fig)
        plt.close(fig)

    maior_classe = counts_pg.idxmax()
    interpretacao(
        f"A classe mais frequente é **{maior_classe}**, com {counts_pg.max():,} menções "
        f"({100*counts_pg.max()/len(df_pg):.1f}% do total). As três classes ficaram "
        f"{'bem equilibradas' if counts_pg.max()/max(counts_pg.min(),1) < 1.5 else 'desbalanceadas'} entre si "
        f"(a maior tem {counts_pg.max()/max(counts_pg.min(),1):.1f}x o volume da menor)."
    )

    st.subheader("Distribuição do Compound Score" + (" (VADER)" if IDIOMA == "en" else ""))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(df_pg["compound"], bins=80, color="#9b59b6", edgecolor="white", linewidth=0.3)
    ax.axvline(0.05, color="#2ecc71", linestyle="--", label="Limiar Positivo (0.05)")
    ax.axvline(-0.05, color="#e74c3c", linestyle="--", label="Limiar Negativo (-0.05)")
    ax.axvline(df_pg["compound"].mean(), color="orange", linestyle=":",
               label=f"Média ({df_pg['compound'].mean():.3f})")
    ax.legend()
    ax.set_xlabel("Compound Score")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)
    plt.close(fig)

    pct_neutro_zero = 100 * (df_pg["compound"].abs() < 0.05).sum() / len(df_pg)
    interpretacao(
        f"O compound score do VADER vai de -1 (bem negativo) a +1 (bem positivo). Aqui, {pct_neutro_zero:.1f}% "
        f"dos textos ficam bem perto de zero (entre -0.05 e 0.05), ou seja, são tratados como neutros — "
        f"comum em textos curtos como tweets, que têm poucas palavras carregadas de sentimento. A média "
        f"geral é {df_pg['compound'].mean():.3f}, "
        f"{'levemente positiva' if df_pg['compound'].mean() > 0.02 else 'levemente negativa' if df_pg['compound'].mean() < -0.02 else 'praticamente neutra'}."
    )

# ----------------------------------------------------------------
# PÁGINA: PRÉ-PROCESSAMENTO
# ----------------------------------------------------------------
elif pagina == "🧪 Pré-processamento":
    st.title("🧪 Pipeline de Limpeza de Texto")
    if IDIOMA == "pt":
        st.markdown(
            "Cada texto passa por: minúsculas → remoção de URLs, @menções, hashtags, "
            "pontuação e números → remoção de *stopwords* em português (mantendo os "
            "acentos, que são importantes no idioma) → tokens com mais de 2 letras."
        )
    else:
        st.markdown(
            "Cada tweet passa por: minúsculas → remoção de URLs, @menções, "
            "hashtags, acentuação/caracteres não-ASCII, pontuação e números → "
            "remoção de *stopwords* em inglês → tokens com mais de 2 letras."
        )
    st.subheader("Antes → Depois (amostra do dataset)")
    amostra = df.sample(6, random_state=SEED)[["texto_original", "clean_text", "sentiment"]]
    for _, row in amostra.iterrows():
        st.markdown(f"**Original:** {row['texto_original']}")
        st.markdown(f"**Limpo:** `{row['clean_text']}`  {sentiment_badge(row['sentiment'])}", unsafe_allow_html=True)
        st.divider()

    st.subheader("Limpeza aplicada na 'Frase para testar' da barra lateral")
    if frase and frase.strip():
        st.markdown(f"**Original:** {frase}")
        st.code(limpar_texto(frase, IDIOMA), language=None)
    else:
        st.caption("Digite algo no campo 'Frase para testar', na barra lateral, para ver o resultado aqui.")

# ----------------------------------------------------------------
# PÁGINA: ML CLÁSSICO
# ----------------------------------------------------------------
elif pagina == "🤖 ML Clássico":
    st.title("🤖 Machine Learning Clássico")
    st.caption("Regressão Logística e Naive Bayes sobre TF-IDF, treinados na amostra carregada.")

    resultados_ml = train_ml_models(df, cache_key=f"{IDIOMA}-{_arquivo_assinatura(DATA_PATH)}")
    st.session_state["resultados_ml"] = resultados_ml

    tabela = pd.DataFrame({
        nome: {
            "Acurácia": f"{r['accuracy']:.3f}",
            "F1 (macro)": f"{r['f1_macro']:.3f}",
            "Tempo treino (s)": f"{r['tempo_treino']:.2f}",
            "Tempo inferência (ms/texto)": f"{r['tempo_infer_ms']:.3f}",
        }
        for nome, r in resultados_ml.items()
    }).T
    st.dataframe(tabela, use_container_width=True)

    melhor_modelo = max(resultados_ml, key=lambda n: resultados_ml[n]["f1_macro"])
    interpretacao(
        f"Nesta amostra, o **{melhor_modelo}** teve o melhor desempenho "
        f"(F1 macro de {resultados_ml[melhor_modelo]['f1_macro']:.3f} e acurácia de "
        f"{resultados_ml[melhor_modelo]['accuracy']:.3f}). F1 macro é a métrica mais justa aqui porque "
        f"trata as três classes (Positivo/Negativo/Neutro) com o mesmo peso, mesmo que uma tenha mais "
        f"exemplos que as outras."
    )

    cols = st.columns(len(resultados_ml))
    for col, (nome, r) in zip(cols, resultados_ml.items()):
        with col:
            plot_confusion(r["y_test"], r["y_pred"], labels_ordenados, nome)
            interpretar_confusao(r["y_test"], r["y_pred"], labels_ordenados, nome)

    st.caption(
        "Como ler a matriz de confusão: a diagonal (canto superior esquerdo ao inferior direito) mostra "
        "os acertos — quanto mais escura e concentrada ali, melhor. Números fora da diagonal são erros: "
        "linha = sentimento real, coluna = sentimento que o modelo previu."
    )

    st.divider()
    st.subheader("🔮 Predição para a 'Frase para testar' da barra lateral")
    if frase and frase.strip():
        st.markdown(f"**Frase:** {frase}")
        for nome, r in resultados_ml.items():
            pred = r["pipeline"].predict([frase])[0]
            st.markdown(f"**{nome}:** {sentiment_badge(pred)}", unsafe_allow_html=True)
    else:
        st.caption("Digite algo no campo 'Frase para testar', na barra lateral, para ver a predição aqui.")

# ----------------------------------------------------------------
# PÁGINA: DEEP LEARNING (LLM)
# ----------------------------------------------------------------
elif pagina == "🧠 Deep Learning (LLM)":
    st.title("🧠 Deep Learning — Modelo Transformer")
    st.markdown(
        f"Usa o modelo `{LLM_MODEL_NAME}` (multilíngue, baseado em DistilBERT) "
        "via `transformers`. **Baixa o modelo na primeira execução (~500MB) e é "
        "mais lento** que o ML clássico — por isso roda só quando você clicar no botão."
    )

    n_amostra = st.slider("Tamanho da amostra a classificar", 20, 500, 200, step=20)

    if st.button("🚀 Rodar modelo de Deep Learning"):
        try:
            with st.spinner("Carregando modelo (pode demorar na primeira vez)..."):
                resultado_llm = run_llm_eval(df, n_amostra=n_amostra)
            st.session_state["resultado_llm"] = resultado_llm
            st.success("Classificação concluída!")
        except Exception as e:
            st.error(
                "Não foi possível carregar/rodar o modelo de Deep Learning "
                f"(`{LLM_MODEL_NAME}`). Verifique se `transformers` e `torch` "
                f"estão instalados e se há acesso à internet.\n\nErro: {e}"
            )

    if "resultado_llm" in st.session_state:
        r = st.session_state["resultado_llm"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Acurácia", f"{r['accuracy']:.3f}")
        c2.metric("F1 (macro)", f"{r['f1_macro']:.3f}")
        c3.metric("Tempo/inferência", f"{r['tempo_infer_ms']:.1f} ms")

        col_grafico, col_vazia = st.columns([1, 1.3])
        with col_grafico:
            plot_confusion(r["y_test"], r["y_pred"], labels_ordenados,
                            f"LLM — amostra de {r['n_amostra']}", figsize=(3.2, 2.8))
        interpretar_confusao(r["y_test"], r["y_pred"], labels_ordenados, "modelo de Deep Learning")

        if "resultados_ml" in st.session_state:
            melhor_ml = max(st.session_state["resultados_ml"],
                             key=lambda n: st.session_state["resultados_ml"][n]["f1_macro"])
            f1_ml = st.session_state["resultados_ml"][melhor_ml]["f1_macro"]
            diff = r["f1_macro"] - f1_ml
            if diff > 0.02:
                comparativo = f"superou o melhor modelo de ML clássico ({melhor_ml}) em {diff:.3f} de F1 macro"
            elif diff < -0.02:
                comparativo = f"ficou {abs(diff):.3f} de F1 macro **abaixo** do melhor modelo de ML clássico ({melhor_ml})"
            else:
                comparativo = f"ficou praticamente empatado com o melhor modelo de ML clássico ({melhor_ml})"
            interpretacao(
                f"Nesta amostra de {r['n_amostra']} textos, o modelo de Deep Learning {comparativo}, "
                f"mas levou {r['tempo_infer_ms']:.1f} ms por texto contra poucos milissegundos do ML "
                f"clássico — a troca é precisão/contexto vs. velocidade/custo computacional."
            )
    else:
        st.info("Clique no botão acima para rodar a avaliação do modelo de Deep Learning (com uma amostra do dataset).")

    st.divider()
    st.subheader("🔮 Predição para a 'Frase para testar' da barra lateral")
    st.caption(
        "Isto é separado da avaliação acima: classifica só a frase digitada, sem precisar rodar "
        "a amostra inteira. Ainda baixa o modelo (~500MB) na primeira vez que você clicar."
    )
    if frase and frase.strip():
        if st.button("🔮 Classificar frase com Deep Learning"):
            try:
                with st.spinner("Carregando modelo (pode demorar na primeira vez)..."):
                    clf = load_llm_pipeline()
                    saida = clf(frase[:512])[0]
                pred_llm = LLM_LABEL_MAP.get(saida["label"].lower(), saida["label"])
                st.session_state["pred_llm_frase"] = (frase, pred_llm, saida["score"])
            except Exception as e:
                st.error(
                    "Não foi possível carregar/rodar o modelo de Deep Learning. Verifique se "
                    f"`transformers` e `torch` estão instalados e se há acesso à internet.\n\nErro: {e}"
                )
        if "pred_llm_frase" in st.session_state:
            frase_classificada, pred_llm, confianca = st.session_state["pred_llm_frase"]
            st.markdown(f"**Frase:** {frase_classificada}")
            st.markdown(f"{sentiment_badge(pred_llm)} — confiança: `{confianca:.3f}`", unsafe_allow_html=True)
            if frase_classificada != frase:
                st.caption("⚠️ A frase mudou desde a última classificação — clique no botão de novo para atualizar.")
    else:
        st.caption("Digite algo no campo 'Frase para testar', na barra lateral, para habilitar o botão.")

# ----------------------------------------------------------------
# PÁGINA: COMPARAÇÃO DE MODELOS
# ----------------------------------------------------------------
elif pagina == "⚖️ Comparação de Modelos":
    st.title("⚖️ Comparação: ML Clássico x Deep Learning")

    linhas = []
    if "resultados_ml" in st.session_state:
        for nome, r in st.session_state["resultados_ml"].items():
            linhas.append({"Modelo": nome, "Acurácia": r["accuracy"], "F1 (macro)": r["f1_macro"],
                            "Tempo/inferência (ms)": r["tempo_infer_ms"]})
    else:
        st.warning("Abra a página **🤖 ML Clássico** primeiro para treinar esses modelos.")

    if "resultado_llm" in st.session_state:
        r = st.session_state["resultado_llm"]
        linhas.append({"Modelo": "Deep Learning (Transformer)", "Acurácia": r["accuracy"],
                        "F1 (macro)": r["f1_macro"], "Tempo/inferência (ms)": r["tempo_infer_ms"]})
    else:
        st.info("Rode a página **🧠 Deep Learning (LLM)** para incluí-lo nesta comparação.")

    if linhas:
        comp = pd.DataFrame(linhas)
        st.dataframe(comp.round(3), use_container_width=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        axes[0].bar(comp["Modelo"], comp["F1 (macro)"], color="#3498db")
        axes[0].set_title("F1 (macro)")
        axes[0].tick_params(axis="x", rotation=20)

        axes[1].bar(comp["Modelo"], comp["Tempo/inferência (ms)"], color="#e67e22")
        axes[1].set_title("Tempo de inferência (ms/texto)")
        axes[1].set_yscale("log")
        axes[1].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        melhor_f1 = comp.loc[comp["F1 (macro)"].idxmax()]
        mais_rapido = comp.loc[comp["Tempo/inferência (ms)"].idxmin()]
        razao_velocidade = comp["Tempo/inferência (ms)"].max() / comp["Tempo/inferência (ms)"].min()
        interpretacao(
            f"O modelo com melhor F1 macro foi **{melhor_f1['Modelo']}** ({melhor_f1['F1 (macro)']:.3f}). "
            f"O mais rápido foi **{mais_rapido['Modelo']}** "
            f"({mais_rapido['Tempo/inferência (ms)']:.2f} ms/texto) — uma diferença de "
            f"{razao_velocidade:.0f}x em relação ao mais lento da comparação. Em produção, a escolha "
            f"depende do volume de menções a processar: para monitorar milhões de menções por dia, "
            f"velocidade costuma pesar mais do que um pequeno ganho de precisão."
        )

    st.divider()
    st.subheader("🔮 Comparação para a 'Frase para testar' da barra lateral")
    st.caption(
        "Aqui você vê, lado a lado, como cada técnica classificaria a MESMA frase — é a forma mais "
        "direta de perceber onde elas concordam e onde divergem."
    )

    if frase and frase.strip():
        st.markdown(f"**Frase:** {frase}")

        if IDIOMA == "pt":
            st.caption(
                "⚠️ O VADER é um léxico feito para **inglês** — em textos em português o resultado "
                "dele tende a ser pouco confiável (geralmente cai em 'Neutro' por não reconhecer as "
                "palavras). Para português, confie mais no ML Clássico/Deep Learning."
            )
        analyzer = get_vader()
        scores = analyzer.polarity_scores(frase)
        compound = scores["compound"]
        vader_label = "Positivo" if compound >= 0.05 else ("Negativo" if compound <= -0.05 else "Neutro")
        st.markdown(
            f"**VADER (léxico):** {sentiment_badge(vader_label)} — compound: `{compound:.3f}`",
            unsafe_allow_html=True,
        )

        if "resultados_ml" in st.session_state:
            for nome, r in st.session_state["resultados_ml"].items():
                pred = r["pipeline"].predict([frase])[0]
                st.markdown(f"**{nome}:** {sentiment_badge(pred)}", unsafe_allow_html=True)
        else:
            st.caption("Abra a página 🤖 ML Clássico primeiro para incluir essa predição aqui.")

        if "resultado_llm" in st.session_state or "pred_llm_frase" in st.session_state:
            try:
                clf = load_llm_pipeline()
                out = clf(frase[:512])[0]
                pred_llm = LLM_LABEL_MAP.get(out["label"].lower(), out["label"])
                st.markdown(
                    f"**Deep Learning (Transformer):** {sentiment_badge(pred_llm)} — confiança: `{out['score']:.3f}`",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass
        else:
            st.caption("Rode a página 🧠 Deep Learning (LLM) pelo menos uma vez para incluir essa predição aqui.")
    else:
        st.caption("Digite algo no campo 'Frase para testar', na barra lateral, para ver a comparação aqui.")

# ----------------------------------------------------------------
# RODAPÉ (aparece no final de qualquer página)
# ----------------------------------------------------------------
render_rodape(idioma_escolhido, dataset_info)
