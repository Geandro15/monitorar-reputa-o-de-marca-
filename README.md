# Monitor de Reputação de Marca — Guia para o Negócio

Painel interativo (Streamlit) que simula o monitoramento de reputação de uma marca nas redes sociais e compara diferentes técnicas de Inteligência Artificial para classificar o sentimento de menções (clientes, comentários, avaliações).

Este documento explica o que foi entregue, para que serve cada parte, e como colocar para rodar — sem exigir conhecimento técnico.

---

## 1. Qual problema isso resolve

Toda marca recebe menções em redes sociais, avaliações e comentários — parte delas elogia, parte reclama, parte só faz perguntas. Acompanhar isso manualmente não escala. Este painel automatiza essa leitura, classificando cada menção como:

| Classificação | O que significa | Uso no negócio |
|---|---|---|
| 🟢 **Positivo** | Elogio, satisfação, recomendação | Identificar embaixadores da marca e casos de sucesso para marketing |
| ⚪ **Neutro** | Menção informativa, dúvida, sem carga emocional forte | Mapear dúvidas frequentes (oportunidade de melhorar FAQ/comunicação) |
| 🔴 **Negativo** | Reclamação, frustração, insatisfação | Sinaliza risco de perda de clientes e de crise de reputação — normalmente pede resposta rápida |

Acompanhar a **proporção** dessas três categorias ao longo do tempo ajuda a perceber uma crise começando antes que ela cresça — por isso o painel tem um alerta automático para dias com muitas menções negativas.

---

## 2. O que foi entregue

| Arquivo / pasta | O que é |
|---|---|
| `app.py` | O painel em si (Streamlit). É o arquivo que você roda. |
| `requirements.txt` | Lista de bibliotecas necessárias para rodar o painel. |
| `tweets_com_sentimentos.csv` | Dataset de exemplo em **inglês** (Sentiment140), tweets reais já classificados. |
| `tweets_com_sentimentos_pt.csv` | Dataset de exemplo em **português**, gerado artificialmente (não são menções reais — ver seção 6). |
| `gerar_dataset_pt.py` | Script que gera o dataset de exemplo em português. Só precisa rodar de novo se quiser recriá-lo. |
| `colab/` | Materiais originais do estudo em Google Colab (notebook, scripts e gráficos que originaram este projeto) — não são usados pelo painel, ficam aqui só como referência/histórico. |

---

## 3. Como rodar (passo a passo)

Pré-requisito: Python instalado no computador.

1. Abra um terminal na pasta do projeto.
2. Crie um ambiente isolado (uma única vez):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Instale as dependências (uma única vez, ou quando o `requirements.txt` mudar):
   ```
   pip install -r requirements.txt
   ```
4. Rode o painel:
   ```
   streamlit run app.py
   ```
5. O navegador abre sozinho em `http://localhost:8501`. Para parar, `Ctrl+C` no terminal.

Nas próximas vezes, basta repetir os passos 2 (ativar) e 4 (rodar) — não precisa reinstalar nada.

---

## 4. Como o painel está organizado

Na barra lateral, você escolhe entre três seções:

### 📖 Início
Página explicativa: o que significam Positivo/Negativo/Neutro para o negócio, e uma comparação das técnicas de IA usadas no painel.

### 📌 Monitoramento da Marca
Simula o acompanhamento de menções reais sobre uma marca/produto. Você digita uma palavra-chave na barra lateral (nome da marca, produto, campanha) e todas as páginas desta seção filtram por ela:

- **Visão Geral** — KPIs (total de menções, % positivas/negativas), gráfico de tendência ao longo do tempo, alertas automáticos de picos negativos, e as menções mais recentes.
- **Buscar Menções** — busca livre por qualquer palavra-chave, com nuvem de palavras e exemplos reais.
- **Nuvens de Palavras** — quais palavras mais aparecem em cada sentimento.
- **Sentimentos & Score** — distribuição estatística dos sentimentos.

### 🧪 Laboratório de IA
Foco técnico: comparar **como** cada modelo de IA decide o sentimento de um texto. Você digita uma frase na barra lateral e ela é classificada em tempo real por diferentes técnicas:

- **Pré-processamento** — mostra como o texto é limpo antes de ser analisado (remoção de links, pontuação, palavras irrelevantes etc.).
- **ML Clássico** — treina e avalia Regressão Logística e Naive Bayes.
- **Deep Learning (LLM)** — avalia um modelo de rede neural (Transformer), mais lento porém mais preciso em textos complexos.
- **Comparação de Modelos** — coloca todas as técnicas lado a lado, tanto em métricas gerais quanto na classificação da frase digitada.

---

## 5. Qual técnica de IA usar?

| Técnica | Velocidade | Precisa treinar? | Entende contexto/ironia? | Quando usar |
|---|---|---|---|---|
| VADER (léxico) | Muito rápida | Não | Não | Triagem rápida em inglês, sem custo de treino |
| Naive Bayes | Muito rápida | Sim (rápido) | Pouco | Grandes volumes, orçamento apertado |
| Regressão Logística | Rápida | Sim (rápido) | Pouco | Bom equilíbrio custo x desempenho |
| Deep Learning (Transformer) | Lenta, exige mais poder computacional | Não (já vem pronto) | Sim, bem melhor | Casos ambíguos ou de alto risco, onde precisão importa mais que velocidade |

Na prática, muitas empresas combinam as duas pontas: uma técnica rápida para triagem geral de todo o volume, e o modelo mais pesado só para os casos que exigem mais confiança.

---

## 6. Limitações importantes (ler antes de apresentar para terceiros)

- **Datas simuladas**: o dataset em inglês é de 2009 e não tem data útil para uma demonstração de "monitoramento recente" — por isso o painel gera datas aleatórias dos últimos 30 dias só para ilustrar o gráfico de tendência. Isso é sinalizado no rodapé de cada página.
- **Dataset em português é sintético**: foi gerado por templates (frases combinadas automaticamente), não são menções reais de clientes. Serve para demonstrar o funcionamento do painel em português. Para uso real, é necessário substituir por um dataset de menções verdadeiras (ver seção 7).
- **VADER não funciona bem em português**: é um dicionário de sentimento feito para inglês. Ao usar o dataset em português, prefira os resultados de ML Clássico/Deep Learning.
- **Todos os dados são de exemplo**: este painel foi construído para demonstrar a capacidade da solução, não para tomar decisões reais de negócio ainda.

---

## 7. Como usar com dados reais da sua marca

O painel foi construído para aceitar qualquer dataset com esta estrutura de colunas: `clean_text`, `sentiment`, `compound`, `pos`, `neg`, `neu`, mais uma coluna com o texto original.

- **Substituir um dos datasets de exemplo**: basta sobrescrever `tweets_com_sentimentos.csv` (ou o `_pt.csv`) com um arquivo real, mantendo o mesmo nome e as mesmas colunas — o painel atualiza sozinho, sem precisar reiniciar nada.
- **Adicionar um terceiro dataset** (ex.: outro idioma, ou uma fonte de dados diferente): exige uma pequena edição no início do arquivo `app.py` (dicionário `DATASETS`).

---

*Dúvidas técnicas ou solicitações de ajuste podem ser encaminhadas para quem configurou este painel.*
