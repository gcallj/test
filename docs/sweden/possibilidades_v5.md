# Possibilidades para uma v5 (investigação de 31/out/2026)

Pedido: investigar outras possibilidades dentro do pedido original:
- estratégia ativa que antecipe altas e quedas;
- poucas operações;
- relevância estatística;
- dados públicos e sentimento de notícias.

Nada aqui foi ligado a retornos. Só medi acesso, cobertura, quantidade de
eventos e custos, para não gastar dados novos antes de congelar uma hipótese
(ADR 0006–0008).

## O que já foi usado

| Dado | Universo | Período ligado a retornos |
|---|---|---|
| Preço e volume (v1–v3) | Large Cap (120) | 2010–2026 |
| Insiders, posições vendidas, tom de balanços/informação privilegiada, reação a balanços (v4) | Large Cap (120) | 2016-07 a 2026-10 |

O que **não** foi usado:
- Mid Cap, Small Cap e First North;
- revisões de analistas e dicas da imprensa;
- recompras, notificações de participação relevante e notícias gerais das
  empresas ("Investor News");
- o comportamento agregado de insiders e vendedores a descoberto;
- qualquer dado depois de 2026-10-30.

## Custos fora das Large Caps

Spread mediano (compra/venda no fechamento de 30/out/2026, screener da Nasdaq):

| Segmento | Ações | Spread mediano | Giro mediano/dia | Giro ≥ 1 MSEK/dia |
|---|---:|---:|---:|---:|
| Large Cap | 163 | 0,19% | 19,6 MSEK | 145 |
| Mid Cap | 139 | 0,51% | 3,1 MSEK | 97 |
| Small Cap | 107 | 1,23% | 0,2 MSEK | 21 |
| First North | 330 | 1,94% | 0,1 MSEK | 32 |

Custo de ida e volta estimado (corretagem 0,25% por ordem + meio spread por lado):
- Mid: ~1,0%;
- Small: ~1,7%.

Com a classe de corretagem zero da Avanza para contas abaixo de 50 kSEK, que
o plano já cita como cenário otimista:
- Mid: ~0,5%;
- Small: ~1,2%.

## Candidatas

| # | Hipótese | Dado | Histórico | Eventos/ano | Expectativa pela literatura | Principal risco |
|---|---|---|---|---:|---|---|
| A | Compras de insiders em **Mid/Small Cap** | registro do FI (já baixado) + preços **incluindo deslistadas** | 2016–2026 | ~230–300 (Mid+Small listadas hoje, ≥ 100–250 kSEK, uma por 60 pregões) | a melhor: estudos suecos acham o efeito em empresas menores | precisa de preço das deslistadas (pago); custo maior |
| B | A mesma A, **para frente** (paper trading) | registro do FI + Yahoo | começa agora | ~230–300 | igual à A | 12 meses para detectar ≥ ~2–3% por evento; só ~60% dos eventos em ações com giro ≥ 1 MSEK |
| C | **Revisões de analistas** e dicas da imprensa | Börsvärlden (sitemap com ~440 mil artigos; ~9.500 de analistas, com hora de publicação) | 2021–2026 denso | ~1.300 (todos os segmentos) | fraca: a revisão chega às corretoras antes da abertura; com 1 dia de atraso quase nada sobra (Alemanha); deriva só em rebaixamentos (EUA) | raspagem de ~10 mil páginas; mapear nome → ticker |
| D | Início de programa de **recompra** | comunicados Nasdaq (já baixados) | 2016–2026 | ~25 nas Large Caps | fraca em large caps; efeito em horizonte de anos | poucos eventos |
| E | **Timing do índice** por insiders ou posições vendidas agregados ("antecipar quedas") | registros do FI | 2016–2026 | 12 decisões/ano | modesta e contestada fora dos EUA (Lakonishok & Lee 2001) | ~20 períodos independentes em 10 anos: sem poder estatístico; já sabemos o que o índice fez |
| F | Tom das notícias gerais ("Investor News": pedidos, aquisições) | comunicados Nasdaq + FinBERT | 2016–2026 | ~1.000 | fraca depois do H3 da v4 | mesmo texto corporativo otimista |
| G | Notificações de participação relevante | comunicados Nasdaq | 2016–2026 | ~60 | quase todas de fundos passivos (BlackRock cruzando 5%) | pouca informação |
| H | Executar no mesmo dia (rodada antes da abertura) | dados da v4 | — | — | efeito de 1–2 dias (Heston & Sinha), menor que o custo | mesmo dado já usado |

### Por que a A precisa de deslistadas

O Yahoo e a API da Nasdaq não servem histórico de empresas que saíram da
bolsa: Swedish Match, ICA, SAS e Collector vêm vazias. No registro de
insiders, as compras em empresas que não estão listadas hoje são uma fatia
grande dos eventos (Nobina, NetEnt, Com Hem, Kungsleden, Gunnebo…). Muitas
saíram por aquisição com prêmio e outras quebraram, então o viés de
sobrevivência tem direção incerta. Um teste só com as sobreviventes não seria
confiável.

Fontes possíveis de preços de deslistadas:
- **EODHD:** cerca de US$ 20–30/mês, com API de deslistadas por bolsa. A
  cobertura de Estocolmo precisa ser conferida com uma chave; a chave de
  demonstração é recusada.
- **Börsdata:** pago, mas não confirmei se cobre deslistadas.
- **FinBas,** do Swedish House of Finance: só para a comunidade acadêmica
  sueca.

## Recomendação

1. **A, com um mês de dados pagos.** É o teste com mais poder (~2.500 eventos
   em 2016–2026, detecta ~0,7% por evento) e a hipótese com melhor base. Mesmo
   protocolo da v4: dry-run com contagens, especificação congelada e uma
   rodada. O critério 1 usaria custos de Mid/Small.
2. **B, se não quiser pagar.** Congela a mesma especificação hoje e registra os
   sinais todo dia, sem dinheiro, por 12 meses. Também serve de mechanics
   check.
3. **C só como segunda hipótese,** em paralelo à A ou à B. A expectativa é
   fraca, mas o dado é novo e gratuito.

D a H não compensam: têm poucos eventos, nenhum poder estatístico, ou usam
dados que já falharam.

**Escolha e resultado**: você escolheu a C, que virou a v5
(`reports/se_v5/verdict.md`) e foi reprovada. A elevação de recomendação deu
+0,41% por evento (t 1,31), o rebaixamento −0,16% e o preço-alvo elevado
−0,82%. A A e a B continuam em aberto.

## Fontes

- [Börsvärlden, revisões de analistas (exemplo)](https://borsvarlden.com/artiklar/flera-analytiker-hojer-sina-riktkurser-for-sandvik)
- [EODHD, dados de empresas deslistadas](https://eodhd.com/financial-apis/landing-delisted/)
- [Swedish House of Finance, FinBas](https://hhs.se/en/houseoffinance/data-center/finbas-stocks)
- [Soucek & Wasserek, recomendações na Alemanha](https://econpapers.repec.org/RePEc:zbw:euvwdp:358)
- [Lakonishok & Lee e evidência agregada (resumos)](https://www.insidearbitrage.com/academic-research-related-to-insider-trading/)
- [Insiders na Suécia, small vs large caps (Gotemburgo)](https://gupea.ub.gu.se/items/d213ae81-4b0c-4240-a522-e913d255bc93)
