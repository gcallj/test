# Conclusão — mercado sueco (1/nov/2026)

Sete estratégias ativas foram desenhadas e julgadas fora da amostra, cada
uma com especificação congelada antes de uma única rodada. Nenhuma superou
o fundo de índice (`^OMXSBGI`, com dividendos) depois dos custos de uma ISK.
A decisão é investir num fundo de índice amplo (ADR 0012).

## O que foi testado

| Versão | Ideia | Melhor número | Por que falhou |
|---|---|---|---|
| v1 | swing por GA, 38 genes, por ticker | −26,2% em 3 anos | trades de 4 dias: ganho bruto +0,42% contra 0,70% de custo |
| v2 | rotação mensal por momentum 12-1 | +35,7% vs +63,6% | momentum fraco no período; menor drawdown |
| v3 | previsão semanal (reversão + momentum) | IC 0,004 (t 0,35) | o sinal do desenvolvimento sumiu fora da amostra |
| v4 | insiders, posições vendidas, tom (FinBERT), balanços | −1,2% a +0,05% por evento | sem retorno anormal nas Large Caps |
| v5 | revisões de analistas (Börsvärlden) | +0,41% por elevação (t 1,31) | a notícia chega depois da reação do preço |
| v6 | swing estilo v1 em conta única, CMA-ES, 13 entradas | +54,5% vs +63,6% | vantagem decai fold a fold; pesos trocam de sinal |
| v7 | calendário: Halloween e mês do dividendo | inverno − verão +0,66%/mês (t 0,91) | sem poder estatístico; ficar fora no verão custou caro |

## Lições para quem continuar

1. **Custo domina.** Com 0,70% de ida e volta, qualquer sinal precisa render
   bem mais que isso por trade. Só prazos de semanas (v6: 32 pregões, +2,65%
   bruto) chegam perto.
2. **Ficar fora do mercado custa caro.** v1 (48% investida) e T1 da v7 (fora
   no verão) perderam mais pelo tempo em caixa que pelos erros de sinal. Vagas
   livres devem ficar no fundo de índice.
3. **Otimizar sobre o mesmo histórico fabrica resultados.** Os pesos da v6
   trocam de sinal entre janelas; a v3 tinha t 3,2 no desenvolvimento e 0,35
   fora dele. Congelar antes, rodar uma vez.
4. **Informação pública já está no preço das Large Caps.** Insiders,
   vendedores a descoberto, tom e analistas: nenhum efeito sobreviveu ao
   controle por quintil de retorno de 63 dias.
5. **Viés de sobrevivência** infla tudo que compra quedas no universo atual.
   Small caps sem preços de deslistadas não dão teste confiável.
6. **Dry-run com contagens** achou bugs de dados reais (tipo de instrumento
   vazio no FI até 2018, detentores com grafias diferentes, LEIs trocados)
   sem gastar a rodada única.

## O que ainda não foi usado

- Dados futuros: paper trading da v6 ou da T1 (v7) congeladas.
- Mid/Small Cap com preços de empresas deslistadas (pago): compras de
  insiders têm a melhor evidência sueca.

Detalhes em `docs/sweden/possibilidades_v5.md` e nos relatórios
`reports/se_v2` a `reports/se_v7`.

## Como investir (a decisão)

1. **Conta**: abra uma ISK numa corretora online (Avanza ou Nordnet). A ISK
   cobra só a taxa anual sobre o patrimônio (schablonskatt), sem imposto por
   venda nem por dividendo.
2. **Fundo**: um fundo de índice sueco amplo, de taxa baixa, que reinveste
   dividendos (acumulante). Opções em nov/2026 (confira a taxa atual na
   página do fundo antes de comprar):
   - **Avanza Zero** — taxa de administração 0%, só na Avanza.
   - **Avanza Sweden** — lançado em maio/2026, ~200 empresas (99% da bolsa),
     0,17% de custo total, só na Avanza.
   - **Länsförsäkringar Sverige Indexnära** — ~0,15%, inclui mid e small cap,
     disponível na Nordnet e na Avanza.
   - **Se a conta for na Nordea**: **Nordea Sverige Passiv** (versão
     acumulante, "icke-utd."), que segue o OMX Stockholm Benchmark Cap
     (~0,19–0,25% ao ano; a Nordea cortou as taxas dos indexnära para 0,19%
     em 2025, confirme no factsheet). Para diversificar: **Nordea Global
     Passiv** (~0,19%). Evite os fundos ativos da Nordea (1–1,5% ao ano): a
     diferença de taxa acumulada em décadas é grande.
3. **Compra**: fundos não pagam courtage. Compre com o valor inteiro (30.000
   SEK) ou em parcelas mensais se preferir suavizar o ponto de entrada.
4. **Manutenção**: nenhuma. Não troque de fundo por desempenho recente, não
   saia no verão. Aportes novos vão para o mesmo fundo.
5. **Diversificação** (opcional, fora do escopo testado): a Suécia é ~1% do
   mercado mundial. Um fundo global de índice (por exemplo Avanza Global ou
   Länsförsäkringar Global Indexnära) ao lado do sueco reduz o risco de um
   único país.

Isto é uma conclusão de pesquisa sobre dados históricos, não aconselhamento
financeiro personalizado.
