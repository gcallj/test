# Pipeline STOCK → BIN → REG → FINAL → GA

Este repositório contém notebooks que devem ser executados **em ordem** para gerar os artefatos finais.


## Testes automatizados (GitHub Actions)

Foi adicionada uma pipeline de testes em `.github/workflows/test.yml` focada em validar apenas o notebook `GA_stock.ipynb`:
- existência do arquivo;
- estrutura básica de notebook (`nbformat`, `cells`);
- presença de pelo menos uma célula de código.
- contrato de I/O do GA: simulação de leitura do `history_consolidated.csv` (schema mínimo) e geração dos arquivos de saída (`.xlsx` e `.csv`) em diretório temporário.
- se existir `history_consolidated.csv` na raiz do repositório, o teste também valida o schema mínimo real do arquivo.
- para evitar versionar arquivo grande (ex.: 165MB), o workflow aceita `secrets.HISTORY_CSV_URL` e baixa o CSV em tempo de execução do GitHub Actions.

Para rodar localmente:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Ordem obrigatória de execução

1. **stock** → `Copy_of_STOCK_ETL_v2.ipynb`
2. **bin** → `bin_Stock_modelos_individuais.ipynb`
3. **reg** → `reg_Stock_modelos_individuais.ipynb`
4. **final** → `Final_stock_output.ipynb`
5. **GA** → `GA_stock.ipynb`

---

## 1) STOCK (`Copy_of_STOCK_ETL_v2.ipynb`)

### Objetivo
Construir o dataset base com OHLCV, features e targets.

### Arquivos gerados
- `expanded_stock.parquet`
- `expanded_stock_reduced.parquet`

### Colunas (estrutura)
O dataset é salvo em **colunas MultiIndex** (`nível 0 = nome da variável`, `nível 1 = ticker`).

#### Bloco OHLCV (por ticker)
- `Open`: abertura do dia.
- `High`: máxima do dia.
- `Low`: mínima do dia.
- `Close`: fechamento do dia.
- `Adj Close`: fechamento ajustado.
- `Volume`: volume negociado.

#### Bloco de targets (por ticker)
- `target_up20`: 1 se houve alta de +20% no horizonte; 0 caso contrário.
- `target_dd5`: 1 se houve queda de -5% no horizonte; 0 caso contrário.
- `target_best_entry`: alvo contínuo de melhor entrada (regressão).
- `target_best_sale`: alvo contínuo de melhor saída/venda (regressão).

#### Bloco de features (por ticker)
- Conjunto de features técnicas/fundamentais/derivadas usadas no treinamento.
- Como é pipeline de engenharia de atributos, o conjunto final pode variar conforme parâmetros do notebook.

---

## 2) BIN (`bin_Stock_modelos_individuais.ipynb`)

### Objetivo
Treinar e aplicar modelos binários para sinais de alta/queda.

### Inputs
- `expanded_stock_reduced.parquet`.

### Arquivos gerados e colunas

## 2.1 `apply_ensemble_signals.csv`
(colunas de `out_apply`)

- `Date`: data de referência.
- `ticker`: ativo.
- `split`: sempre `APPLY` neste arquivo.
- `p_up20`: probabilidade prevista de alta (+20%).
- `p_dd5`: probabilidade prevista de drawdown (-5%).
- `thr_up20`: threshold usado para classificar `up20`.
- `thr_dd5`: threshold usado para classificar `dd5`.
- `pred_up20`: classe prevista para alta (0/1).
- `pred_dd5`: classe prevista para drawdown (0/1).
- `buy`: flag de compra (`pred_up20=1` e `pred_dd5=0`).
- `sell`: flag de venda (`pred_up20=0` e `pred_dd5=1`).
- `action`: ação final (`BUY`, `SELL`, `HOLD`).
- `buy_trust`: confiança da compra (`p_up20*(1-p_dd5)`).
- `sell_trust`: confiança da venda (`(1-p_up20)*p_dd5`).
- `action_trust`: confiança associada à ação final.
- `y_up20`: label real de `up20` (no APPLY tende a `NaN`).
- `y_dd5`: label real de `dd5` (no APPLY tende a `NaN`).
- `trust_up20`: confiança do classificador para `up20`.
- `margin_up20`: margem para o threshold de `up20`.
- `trust_dd5`: confiança do classificador para `dd5`.
- `margin_dd5`: margem para o threshold de `dd5`.

## 2.2 `apply_ensemble_signals_debug.csv`
- Mesmas colunas acima, mas com linhas de `VALID + APPLY` para depuração.

## 2.3 `ensemble_signals_history.parquet`
Arquivo histórico minimizado (função `save_hist_out_minimal`) com colunas:
- `index`: índice sequencial.
- `Date`: data.
- `ticker`: ativo.
- `split`: partição (`VALID`, `TEST`, `FINAL`, `APPLY`, etc. conforme execução).
- `p_up20`: probabilidade de alta.
- `p_dd5`: probabilidade de drawdown.

## 2.4 `ensemble_signals_history_with_trust.parquet`
Histórico enriquecido com colunas de confiança por modelo (`buy_trust__*`, `sell_trust__*`, `action_trust__*`) além das colunas-base do histórico.

---

## 3) REG (`reg_Stock_modelos_individuais.ipynb`)

### Objetivo
Gerar previsões contínuas de preço (compra/venda) com bandas de erro.

### Arquivos gerados e colunas

## 3.1 `apply_forecast_with_error_band.csv`
(saída `APPLY_VIEW`)

- `Date`: data.
- `ticker`: ativo.
- `target`: alvo (`target_best_entry` ou `target_best_sale`).
- `asset_class`: classe do ativo.
- `model`: identificação do baseline/calibração.
- `used_ml`: flag de uso de ML.
- `alpha`: parâmetro do modelo.
- `band_q_lo`: quantil configurado inferior da banda.
- `band_q_hi`: quantil configurado superior da banda.
- `resid_q_lo`: quantil residual inferior estimado.
- `resid_q_hi`: quantil residual superior estimado.
- `resid_fit_n`: nº de pontos usados no fit residual.
- `band_scale`: escala da banda.
- `band_scale_fallback`: se usou fallback de banda.
- `cap_lo`: limite inferior de cap.
- `cap_hi`: limite superior de cap.
- `pred_pct_center`: previsão central (%) após cap.
- `pred_pct_lo_uncapped`: banda inferior (%) sem cap.
- `pred_pct_hi_uncapped`: banda superior (%) sem cap.
- `pred_pct_lo_capped`: banda inferior (%) com cap.
- `pred_pct_hi_capped`: banda superior (%) com cap.
- `op_pct`: percentual operacional final.
- `y_true_pct`: valor real (%) quando disponível.
- `pred_price_center`: preço previsto central.
- `pred_price_lo_uncapped`: preço inferior sem cap.
- `pred_price_hi_uncapped`: preço superior sem cap.
- `pred_price_lo_capped`: preço inferior com cap.
- `pred_price_hi_capped`: preço superior com cap.
- `op_price`: preço operacional final.
- `abs_err_pct`: erro absoluto percentual.
- `mae_valid`, `p95_valid`, `covU_valid`, `covC_valid`: métricas de validação.
- `mae_test`, `p95_test`, `covU_test`, `covC_test`: métricas de teste.
- `touchC_valid`, `touchOp_valid`, `touchC_test`, `touchOp_test`: métricas de toque/cobertura.

## 3.2 `forecast_history_wide.parquet` / `forecast_history_wide.csv`
(colunas fixas: `HISTORY_WIDE_COLS`)

- `Date`: data.
- `ticker`: ativo.
- `best_buy_price`: melhor preço projetado para compra.
- `buy_err_lo`: limite inferior de erro (compra).
- `buy_err_hi`: limite superior de erro (compra).
- `best_sell_price`: melhor preço projetado para venda.
- `sell_err_lo`: limite inferior de erro (venda).
- `sell_err_hi`: limite superior de erro (venda).

## 3.3 `baseline_best_by_ticker_target.csv`
Resumo dos melhores baselines por par `(ticker, target)` (colunas de identificação e hiperparâmetros/score do melhor baseline).

## 3.4 `band_scale_by_ticker_target.csv`
- `ticker`: ativo.
- `target`: alvo.
- `asset_class`: classe.
- `band_scale`: escala de banda escolhida.
- `fallback_used`: indica fallback na banda.

## 3.5 `valid_compare.csv` e `test_compare.csv`
Comparativos por ticker/target para avaliação (mesmo esquema base de colunas de métricas de erro/cobertura usadas no notebook).

---

## 4) FINAL (`Final_stock_output.ipynb`)

### Objetivo
Consolidar REG + BIN + fundamentos e calcular score final de decisão.

### Inputs diretos
- REG wide: `forecast_history_wide.parquet`.
- BIN: `ensemble_signals_history.parquet`.
- Fundamentais (coletados no notebook).

### Arquivos gerados
- `history_consolidated.parquet`

### Colunas do output consolidado (fixas: `FINAL_COLS`)
- `Date`: data.
- `ticker`: ativo.
- `split`: partição do dado.
- `price`: preço base da linha (no notebook, igual a `close`).
- `open`, `high`, `low`, `close`: OHLC do dia.
- `best_buy_value`: preço alvo de compra (normalizado da REG).
- `best_sell_value`: preço alvo de venda (normalizado da REG).
- `err_buy_pct`: erro percentual da estimativa de compra.
- `err_sell_pct`: erro percentual da estimativa de venda.
- `downside_pct`: risco percentual até o piso estimado.
- `upside_pct`: potencial percentual até o topo estimado.
- `risk_return`: razão risco/retorno.
- `p_up20`: probabilidade de alta (BIN).
- `p_dd5`: probabilidade de drawdown (BIN).
- `pred_up20`: previsão binária de alta.
- `pred_dd5`: previsão binária de drawdown.
- `up20_bin`: classe binária final de alta.
- `dd5_bin`: classe binária final de drawdown.
- `buy_trust`: confiança de compra (derivada de probas).
- `sell_trust`: confiança de venda (derivada de probas).
- `EV_buy_reg`: expected value com componente de regressão.
- `EV_buy_ens`: expected value com componente de ensemble/binário.
- `EV_buy`: EV combinado.
- `fund_score`: score fundamentalista.
- `EV_buy_fund`: EV ajustado por fundamento (versão 1).
- `EV_buy_fund_2`: EV ajustado por fundamento (versão 2).
- `EV_buy_fund_3`: EV ajustado por fundamento (versão 3; usado no GA).
- `signal`: sinal final consolidado.
- `dividend_yield`: dividend yield do ativo.
- `trailing_pe`: preço/lucro.
- `price_to_book`: preço/valor patrimonial.
- `market_cap`: valor de mercado.

---

## 5) GA (`GA_stock.ipynb`)

### Objetivo
Rodar etapa final de seleção/otimização (GA + walk-forward) e gerar relatório/sinais.

### Input
- `history_consolidated.parquet` (usa `Date`, `ticker`, `price` e score, por padrão `EV_buy_fund_3`).

### Arquivos gerados e colunas

## 5.1 Excel
`apply_PER_TICKER_WFGA_intraday__H{FWD_H}__APPLY{APPLY_DAYS}D__v2.xlsx`

### Aba `summary_latest` (colunas)
- `ticker`, `feat_count_used`
- `wf_auc_mean`, `wf_auc_std`, `wf_acc_mean`, `wf_ap_mean`, `wf_logloss`, `wf_brier`
- `ga_enter_abs`, `ga_exit_abs`, `ga_atr_mult`, `ga_rr_mult`
- `ga_return_1y`, `ga_mdd_1y`, `ga_sharpe_1y`, `ga_trades_1y`, `ga_exposure_1y`, `ga_fitness_1y`
- `buyhold_return_1y`
- `test_return`, `test_mdd`, `test_sharpe`, `test_n_trades`
- `use_strategy_flag`
- `latest_date`, `latest_close`, `latest_atr`, `latest_score_z`
- `signal_eod`, `signal`
- `score_0_100`
- `train_start`, `train_end`, `test_start`, `test_end`

### Aba `apply_last_{APPLY_DAYS}d` / CSV `apply_last_{APPLY_DAYS}d__H{FWD_H}__v2.csv`
- `Date`, `ticker`, `close`
- `signal_eod`, `signal`, `use_strategy_flag`
- `score_0_100`, `score_z`
- `wf_auc_mean`, `wf_auc_std`, `wf_acc_mean`, `wf_ap_mean`, `wf_logloss`, `wf_brier`
- `ga_enter_abs`, `ga_exit_abs`, `ga_atr_mult`, `ga_rr_mult`
- `ga_return_1y`, `ga_mdd_1y`, `ga_sharpe_1y`, `ga_trades_1y`, `ga_exposure_1y`
- `buyhold_return_1y`
- `test_return`, `test_mdd`, `test_sharpe`, `test_n_trades`
- `next_day_filled`
- `limit_price_next_day`, `best_buy_value`, `best_sell_value`, `entry_ref_price`
- `stop_abs`, `take_abs`, `stop_pct`, `take_pct`
- `buy_entry`, `buy_stop`, `buy_take`
- `sell_entry`, `sell_stop`, `sell_take`
- `train_start`, `train_end`, `test_start`, `test_end`

---

## Fluxo resumido de arquivos

1. STOCK gera os parquets base (`expanded_stock*`).
2. BIN gera sinais probabilísticos/classificação (`apply_ensemble_signals*`, `ensemble_signals_history*`).
3. REG gera preços de compra/venda com bandas (`apply_forecast_with_error_band`, `forecast_history_wide`).
4. FINAL consolida tudo em `history_consolidated*` com EV e sinais finais.
5. GA gera relatório final (Excel) e CSV de aplicação dos últimos dias.
