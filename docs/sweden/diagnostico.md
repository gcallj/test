# Diagnóstico — por que o sistema de swing trade (v1) não funciona

Medido no replay walk-forward oficial (run 41139471106, 31/out/2026),
`reports/se/`. O sistema perdeu 26,2% em 3 anos contra +63,6% do índice com
dividendos. Os números abaixo mostram que a causa é estrutural, não de ajuste
de parâmetros.

## Evidência

| Medida | Valor | Leitura |
|---|---:|---|
| Ganho bruto médio por trade | +0,42% | é o que o sinal entrega antes de custos |
| Custo por trade (ida e volta) | 0,70% | **166% do ganho bruto**: cada trade médio perde dinheiro |
| Duração média do trade | 4 pregões | giro de ~61× o capital por ano |
| Custo anual estimado | ~21% do capital | 183 trades/ano × 0,70% × 5.000 SEK ÷ 30.000 SEK |
| Exposição média ao mercado | ~48% | metade do tempo fora de um mercado que subiu 17,8% a.a. |
| Ganho médio ÷ perda média | 0,63 | saídas por alvo curto cortam os vencedores |
| Deriva do mercado em 4 pregões | ~0,28% | o ganho bruto (+0,42%) é quase só o mercado subindo |
| Folds com expectancy positiva | 2 de 6 | sem estabilidade entre regimes |
| Parâmetros otimizados | 38 genes | muito espaço para ajustar ruído |

## Causas

1. **Horizonte errado para o custo.** Trades de 4 dias com alvo de 1–2 ATR
   não pagam 0,70% de ida e volta. Nenhum ajuste de genes resolve: o ganho
   bruto por trade teria de quase triplicar.
2. **Fora do mercado metade do tempo.** O prêmio das ações vem de ficar
   investido. Um sistema que fica 50% em caixa precisa de alpha enorme só
   para empatar com o índice.
3. **O otimizador mira a métrica errada.** A fitness do GA maximiza medianas
   de win rate por ticker. O objetivo real é o retorno da carteira contra o
   índice com drawdown menor. Win rate alto com payoff 0,63 não é edge.
4. **Cada ação isolada.** O backtest avalia cada ticker sozinho, com capital
   próprio. As anomalias documentadas em ações (momentum, por exemplo) são
   comparações *entre* ações; o sistema não tinha como vê-las.
5. **Sinal sem fundamento.** 60 features por ticker votam com sinal
   decidido por palavras no nome da coluna ("risk", "down" = baixista). Não há
   evidência de que essa votação prevê retornos de 4 dias.
6. **Superfície de ajuste grande.** 38 genes num GA, treinados em 4 anos,
   produzem genomas que diferem muito entre folds — o que o resultado fora da
   amostra (2 de 6 folds positivos) confirma.

## O que a v2 muda

A estratégia v2 (`rotation_se.py`, ADR 0005) ataca as causas diretamente:

| Causa | v2 |
|---|---|
| Horizonte vs custo | rebalanceamento mensal, ações mantidas por meses; giro ~2–3× ao ano |
| Fora do mercado | sempre investida, exceto quando o filtro de regime manda para o caixa |
| Métrica errada | parâmetros escolhidos pela métrica do critério: excesso sobre o índice com penalidade de drawdown, na carteira |
| Ações isoladas | ranking entre as ações do universo (momentum relativo) |
| Sinal sem fundamento | momentum de 12 meses, uma das anomalias mais documentadas em ações, inclusive nórdicas |
| Superfície de ajuste | 6–8 parâmetros discretos, escolhidos num período de desenvolvimento separado |

## Limites que continuam

- **Viés de sobrevivência**: o universo é a lista Large Cap de hoje. Uma
  estratégia que compra quem subiu se beneficia de saber quem chegou a large
  cap. Os números do período de desenvolvimento (2014–2023) estão inflados por
  isso; os 3 anos de avaliação menos, mas não zero.
- **Período de avaliação já visto**: os 3 anos de avaliação (nov/2023 a
  out/2026) já foram usados para avaliar a v1, e eu sei que o mercado subiu
  forte neles. A v2 foi desenhada e escolhida só com dados até 30/10/2023 e
  avaliada nesses 3 anos uma única vez, mas não dá para "desver" o período.
