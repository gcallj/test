# Dados alternativos para o mercado sueco

Levantamento de 31/out/2026, feito antes de ligar qualquer um destes dados a
retornos. As fontes usadas pela estratégia v4 estão no ADR 0008. O código está
em `altdata_se.py`.

## Por que dados novos

v1, v2 e v3 usaram só preço e volume, e toda a história de 2010 a 2026 já
serviu para desenhar ou julgar essas estratégias. Uma ideia nova que use os
mesmos preços só mede o quanto ela se ajustou ao passado. Os registros
públicos abaixo trazem informação que o preço ainda não tem no momento da
publicação, e nenhuma versão anterior olhou para eles.

## Fontes avaliadas

| Fonte | O que traz | Desde | Quando fica pública | Acesso | Na v4? |
|---|---|---|---|---|---|
| **FI — registro de insiders** (insynshandel, MAR) | cada compra/venda de diretor, conselheiro e pessoas ligadas: data, hora de publicação, natureza, volume, preço, local de negociação, se é programa de ações | jul/2016 | hora exata de publicação | CSV público, no máximo 1.000 linhas por consulta (baixado em janelas) | **H1** |
| **FI — registro de posições vendidas** (blankningsregistret) | posição líquida vendida ≥ 0,5% do capital, por detentor, e o histórico de cada mudança | nov/2012 | posição de T publicada em T+1; usamos a partir de T+2 | planilha ODS pública | **H2** (filtro) |
| **Nasdaq Nordic — comunicados das empresas** | título, categoria (balanço, informação privilegiada…), hora e texto completo | 2005 (por empresa) | hora do comunicado | API JSON pública sem chave (não documentada) | **H3, H4** |
| **FinBERT** (`ProsusAI/finbert`) | probabilidade de tom positivo/negativo/neutro de cada frase em inglês | — | — | modelo aberto, roda em CPU (~100 títulos/s) | **H3** |
| FI — participações dos fundos (fondinnehav) | carteira de cada fundo sueco por trimestre | 2018T4 | desconhecido: arquivos republicados em 2022 | ZIP público | não — sem data de publicação original, o backtest não seria causal |
| GDELT | tom de notícias globais | 2015 | 15 min | API pública | não — cobertura fraca de empresas suecas e difícil ligar a notícia ao ticker |
| Google Trends | interesse de busca relativo | 2004 | semanal | sem API oficial, com limite de requisições | não — ruidoso, mede atenção de varejo |
| Riksbank, SCB, Konjunkturinstitutet | juros, inflação, confiança | décadas | calendário fixo | APIs públicas | não — filtros de mercado pioraram todas as variantes da v2 e da v3 |
| Börsdata, Modular Finance (Holdings), LSEG/Infront | fundamentos trimestrais, donos, estimativas de analistas | variável | variável | **pagos** | não — opção se você quiser pagar |
| MFN, Cision | comunicados (mesmo conteúdo da Nasdaq) | — | — | feeds públicos | não — redundante |

## Cobertura medida (universo de 120 tickers)

| Fonte | Linhas no universo | Tickers | Observações |
|---|---:|---:|---|
| Insiders | 36.890 de 168.226 | 115 | ligação por ISIN atual; ISIN antigo só se LEI e nome do emissor concordarem (o registro tem LEIs de outra empresa digitados por engano) |
| Posições vendidas | 21.901 de 39.261 | 104 | emissores estrangeiros (AstraZeneca, ABB, Stora Enso, Nordea, Lundin…) são supervisionados no país de origem e não aparecem |
| Comunicados | 96.673 | 120 | a Nasdaq rebatiza os comunicados antigos com o nome atual; a Nordea antiga (sueca, até 2018) foi somada à nova; Evolution, Embracer e Paradox têm lacunas do tempo de First North |
| Tom (FinBERT) | ~6.700 comunicados em inglês | — | balanços e informação privilegiada desde jul/2015 (um ano antes do período de teste, para o percentil móvel) |

Nomes de categoria mudaram no início de 2017: "Quarterly report" virou
"Interim report (Q1 and Q3)", e "Inside information" passou a existir como
categoria.

## O que a literatura diz (e o que esperar)

- **Compras de insiders.** Nos EUA, compras de executivos antecedem retorno
  anormal positivo, e as vendas quase nada (Lakonishok & Lee 2001). As compras
  "oportunistas", fora do padrão de calendário de cada pessoa, rendem cerca de
  0,8% ao mês (Cohen, Malloy & Pomorski 2012). Na Suécia, a evidência vem
  quase toda de dissertações, e o resultado é misto: um estudo de 2019–2024
  com o registro do FI não acha retorno positivo depois das compras. O efeito
  costuma ser maior em empresas pequenas, e o nosso universo é Large Cap.
- **Posições vendidas.** Os vendedores grandes divulgados na Europa parecem
  informados: os preços não revertem depois da divulgação (Jones, Reed & Waller
  2016, RFS). O resultado em preços fora das emissões de direitos varia entre
  versões do artigo. Numa ISK não dá para vender a descoberto, então o uso
  prático é evitar os nomes mais vendidos.
- **Tom das notícias.** A notícia de um dia é absorvida em 1–2 dias; agregada
  por semana, prevê até 13 semanas (Heston & Sinha 2017). Comunicados escritos
  pela própria empresa tendem a ser positivos, por isso a v4 compara o tom com
  o do ano anterior em vez de usar um nível fixo.
- **Deriva pós-balanço.** Carteiras ordenadas pela reação de 3 dias ao balanço
  rendem cerca de 12,5% ao ano nos EUA (Brandt et al. 2008). Os estudos
  nórdicos recentes, também quase todos dissertações, acham uma deriva mais
  fraca ou nenhum retorno em carteira.

Expectativa honesta antes de rodar: os efeitos publicados são de mercados
grandes, em empresas menores e com custos institucionais. Com 0,70% de custo
de ida e volta e um universo Large Cap, o mais provável é que no máximo uma
das quatro hipóteses passe.

## Fontes

- Lakonishok & Lee (2001), *Are Insider Trades Informative?*, RFS 14(1).
- [Cohen, Malloy & Pomorski (2012), *Decoding Inside Information*, JF 67(3)](https://www.nber.org/papers/w16454)
- [Jones, Reed & Waller (2016), *Revealing Shorts*, RFS 29(12)](https://econpapers.repec.org/RePEc:oup:rfinst:v:29:y:2016:i:12:p:3278-3320.)
- [Heston & Sinha (2017), *News vs. Sentiment*, FAJ 73(3)](https://www.federalreserve.gov/econres/feds/news-versus-sentiment-predicting-stock-returns-from-news-stories.htm)
- Brandt, Kishore, Santa-Clara & Venkatachalam (2008), *Earnings Announcements Are Full of Surprises*, SSRN 909563.
- Tetlock, Saar-Tsechansky & Macskassy (2008), *More Than Words*, JF 63(3).
- Estudos suecos (dissertações, não revisadas por pares):
  - [Aalto](https://aaltodoc.aalto.fi/items/46b03acc-1857-4858-baf5-5e353e2057e6)
  - [Gotemburgo, EUA vs. Suécia](https://gupea.ub.gu.se/items/f91c060b-18b0-4f43-8966-bdcc6d1a78ed)
  - [Gotemburgo, deriva nórdica 2014–2022](https://gupea.ub.gu.se/items/7563b4c6-c300-4b97-bf57-efc52e859121)
- [Finansinspektionen, blankningsregistret](https://www.fi.se/sv/vara-register/blankningsregistret/)
- [Finansinspektionen, insynshandel](https://marknadssok.fi.se/Publiceringsklient/sv-SE/Search/Search?SearchFunctionType=Insyn)

## Como atualizar

```bash
python altdata_se.py insiders   # registro de insiders (janelas de até 1.000 linhas, com cache)
python altdata_se.py shorts     # posições vendidas (histórico + atuais)
python altdata_se.py news       # comunicados por empresa (mapa em data/news_companies_se.json)
python altdata_se.py tone       # texto dos balanços/informação privilegiada + FinBERT
python altdata_se.py coverage   # cobertura por fonte e ano
```

O FinBERT precisa de `torch` (CPU) e `transformers`. No WSL2:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

```bash
pip install transformers
```
