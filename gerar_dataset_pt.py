"""Gera um dataset SINTETICO de exemplo (mencoes de marca em portugues)
para demonstrar o app em portugues, ja que nao existe um dataset real em
PT anexado ao projeto (o dataset real usado no notebook Colab precisa ser
baixado via API do Kaggle com credenciais).
"""
import random
import re
import string
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

produtos = [
    "o produto", "o pedido", "a compra", "o celular", "o tênis", "a blusa",
    "o notebook", "o fone de ouvido", "o eletrodoméstico", "a bolsa",
    "o perfume", "o livro", "a bicicleta", "o relógio", "o brinquedo",
]
aspectos = [
    "o atendimento", "a entrega", "o suporte", "a qualidade", "o preço",
    "o app", "o site", "a embalagem", "o prazo de entrega", "a experiência de compra",
]
elogios = [
    "rápido e eficiente", "muito profissional", "acima da expectativa",
    "impecável", "super atencioso", "de primeira qualidade",
]
reclamacoes = [
    "demorou semanas para resolver", "ninguém respondeu meus e-mails",
    "veio tudo errado", "cobraram duas vezes no cartão",
    "o produto quebrou em uma semana", "a caixa chegou toda amassada",
]

positivos = [
    "Adorei {aspecto} da marca, muito {elogio}!",
    "Comprei {produto} e chegou rápido, super satisfeito!",
    "Excelente {aspecto}, recomendo a todos!",
    "Melhor experiência que já tive com {aspecto}.",
    "{produto} superou minhas expectativas, muito bom!",
    "Atendimento nota 10, resolveram meu problema rapidinho.",
    "Voltarei a comprar com certeza, {elogio}!",
    "{aspecto} impecável, virei cliente fiel dessa marca.",
    "Muito satisfeito com {produto}, vale cada centavo.",
    "Equipe {elogio}, super recomendo essa marca!",
    "Ótimo {aspecto}, estou muito feliz com a compra!",
    "{produto} é maravilhoso, adorei demais!",
    "Simplesmente incrível, {aspecto} sensacional!",
    "Muito bom {produto}, ficou perfeito, adorei o resultado!",
    "Que ótima experiência de compra, tudo maravilhoso!",
]
negativos = [
    "Péssimo {aspecto}, não recomendo.",
    "{produto} veio com defeito, muito decepcionado.",
    "Atendimento horrível, ninguém resolve nada.",
    "Nunca mais compro dessa marca, {reclamacao}.",
    "Entrega atrasou muito, {reclamacao}.",
    "Produto de baixa qualidade, dinheiro jogado fora.",
    "Suporte não responde, estou muito insatisfeito.",
    "{aspecto} horrível, {reclamacao}.",
    "Extremamente decepcionado com {produto}.",
    "Não indico essa marca pra ninguém, {reclamacao}.",
    "Muito ruim {aspecto}, experiência péssima.",
    "{produto} é horrível, não vale o preço, decepção total.",
    "Que compra ruim, {produto} veio quebrado.",
    "Serviço ruim, atendimento péssimo, não voltarei.",
    "Terrível, simplesmente terrível essa experiência.",
]
neutros = [
    "Comprei {produto} ontem, ainda vou testar.",
    "Alguém sabe informar o prazo de entrega de {produto}?",
    "Recebi {produto}, embalagem estava ok.",
    "Fui até a loja física para trocar {produto}.",
    "{produto} chegou no prazo previsto.",
    "Vi a propaganda da marca na TV hoje.",
    "Qual a diferença entre {produto} e o modelo anterior?",
    "Estou pesquisando opiniões sobre {aspecto} antes de comprar.",
    "{produto} tem cor azul e branca, conforme o anúncio.",
    "A loja fica aberta até as 22h, segundo o site.",
]

STOPWORDS_PT = set("""
a à às ao aos as com como da das de dela dele deles delas do dos e é essa
essas esse esses esta estas este estes eu foi for foram fosse isso isto já
lhe lhes mais mas me mesmo meu meus minha minhas muito na nas nem no nos
nossa nossas nosso nossos num numa não o os ou para pela pelas pelo pelos
por qual quando que quem se seu seus sua suas só também te tem tinha
tive tu tua tuas tém têm um uma umas uns você vocês
""".split())


def limpar_pt(texto):
    t = texto.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\d+", "", t)
    tokens = [w for w in t.split() if w not in STOPWORDS_PT and len(w) > 2]
    return " ".join(tokens)


def gerar_frase(template):
    return template.format(
        produto=random.choice(produtos),
        aspecto=random.choice(aspectos),
        elogio=random.choice(elogios),
        reclamacao=random.choice(reclamacoes),
    )


N_POR_CLASSE = 1500
linhas = []
for templates, label, faixa in [
    (positivos, "Positivo", (0.4, 0.9)),
    (negativos, "Negativo", (-0.9, -0.4)),
    (neutros, "Neutro", (-0.05, 0.05)),
]:
    for _ in range(N_POR_CLASSE):
        template = random.choice(templates)
        texto = gerar_frase(template)
        texto = texto[0].upper() + texto[1:]
        compound = float(np.clip(np.random.uniform(*faixa), -1, 1))
        pos = max(0.0, compound) * random.uniform(0.6, 1.0)
        neg = max(0.0, -compound) * random.uniform(0.6, 1.0)
        neu = max(0.0, 1 - pos - neg)
        linhas.append({
            "5": texto,
            "clean_text": limpar_pt(texto),
            "sentiment": label,
            "compound": round(compound, 4),
            "pos": round(pos, 3),
            "neg": round(neg, 3),
            "neu": round(neu, 3),
        })

df = pd.DataFrame(linhas)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # embaralha

df.to_csv("tweets_com_sentimentos_pt.csv", index=False)
print("Gerado:", df.shape)
print(df["sentiment"].value_counts())
print(df.head(5))
