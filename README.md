# Fluxo do projeto de modelos de estoque

Este repositório contém notebooks Jupyter para preparação de dados, treinamento de modelos e geração de saídas finais relacionadas a previsões/classificações de estoque.

## Ordem sugerida de execução

1. **`Copy of STOCK_ETL_v2.ipynb`**
   - **Objetivo:** realizar o ETL (extração, transformação e carga) dos dados de estoque.
   - **Resultado esperado:** datasets tratados e prontos para o treinamento dos modelos.

2. **`bin_Stock_modelos_individuais.ipynb`**
   - **Objetivo:** treinar e avaliar modelos individuais para tarefas **binárias** (ex.: classificação do status de estoque).
   - **Resultado esperado:** métricas e artefatos dos modelos binários.

3. **`reg_Stock_modelos_individuais.ipynb`**
   - **Objetivo:** treinar e avaliar modelos individuais para tarefas de **regressão** (ex.: previsão de quantidade/nível de estoque).
   - **Resultado esperado:** métricas e artefatos dos modelos de regressão.

4. **`bin_Ensemble_stock.ipynb`**
   - **Objetivo:** criar e avaliar o **ensemble** para os modelos binários.
   - **Resultado esperado:** resultados consolidados do ensemble binário.

5. **`Final_stock_output.ipynb`**
   - **Objetivo:** consolidar as previsões finais e gerar o output final do projeto.
   - **Resultado esperado:** dataset/relatório final com as previsões e/ou análises finais.

## Observações

- Execute os notebooks na ordem acima para garantir que os dados e artefatos necessários estejam disponíveis nas etapas seguintes.
- Caso você já possua os dados tratados, é possível iniciar direto nos notebooks de modelos, mas o resultado final pode variar conforme o pré-processamento aplicado.
