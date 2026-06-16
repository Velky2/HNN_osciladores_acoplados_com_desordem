# Otimização de Hiperparâmetros de Redes Neurais Hamiltonianas para Osciladores Acoplados com Desordem

Comparação de três estratégias de otimização de hiperparâmetros (HPO), **Optuna/TPE**, **CMA-ES** e **TuRBO**, aplicadas ao treinamento de **Redes Neurais Hamiltonianas (HNNs)** em um sistema de `N = 8` osciladores não linearmente acoplados com desordem aleatória congelada (intensidade da desordem `σ_J`). O sistema desordenado é um análogo clássico e contínuo do modelo de spin-glass de Sherrington–Kirkpatrick.

## Visão geral

O projeto investiga se a dificuldade do espaço de busca da HPO co-varia com a complexidade física do sistema (controlada por `σ_J`) e se isso afeta qual otimizador apresenta melhor desempenho. Como contribuição secundária, utiliza **SHAP** para interpretabilidade informada pela física em dois níveis:

- **Modelo substituto (XGBoost + TreeExplainer):** quais hiperparâmetros mais influenciam o desempenho da HNN.
- **MLP da HNN (GradientExplainer):** quais variáveis físicas dominam o Hamiltoniano aprendido.

Os dados de treinamento são gerados com um integrador **simplético de Störmer–Verlet** para evitar deriva secular de energia, e a conservação de energia é verificada para os modelos aprendidos.

## Estrutura do repositório

Execute os notebooks na seguinte ordem:

| # | Notebook | Descrição |
|---|----------|-----------|
| 1 | `hnn.ipynb` | Rede Neural Hamiltoniana: arquitetura, simulação física e treinamento. |
| 2 | `optuna.ipynb` | HPO com Optuna (sampler TPE). |
| 3 | `turbo.ipynb` | HPO com TuRBO (Otimização Bayesiana com Regiões de Confiança, via BoTorch). |
| 4 | `cmaes.ipynb` | HPO com CMA-ES. |
| 5 | `analysis.ipynb` | Comparação dos três otimizadores, visualizações e interpretabilidade via SHAP. |

Um módulo compartilhado `common.py` cuida da simulação física e da geração de dados (pré-computados fora da função objetivo da HPO para evitar computação redundante).


## Requisitos

- Python 3.9+
- [PyTorch](https://pytorch.org/) (HNN, gradientes do Hamiltoniano via autograd)
- NumPy, SciPy (integração numérica)
- [Optuna](https://optuna.org/) (TPE)
- [cma](https://github.com/CMA-ES/pycma) (CMA-ES)
- [BoTorch](https://botorch.org/) (TuRBO)
- [XGBoost](https://xgboost.ai/) e [SHAP](https://github.com/shap/shap) (interpretabilidade)
- Jupyter / JupyterLab


## Referências

1. Sherrington, D. & Kirkpatrick, S. (1975). *Solvable model of a spin-glass.* Physical Review Letters, 35(26), 1792–1796.
2. Greydanus, S., Dzamba, M., & Yosinski, J. (2019). *Hamiltonian Neural Networks.* Advances in Neural Information Processing Systems (NeurIPS), 32.
3. Hansen, N. & Ostermeier, A. (2001). *Completely Derandomized Self-Adaptation in Evolution Strategies.* Evolutionary Computation, 9(2), 159–195. (CMA-ES)
4. Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., & Poloczek, M. (2019). *Scalable Global Optimization via Local Bayesian Optimization.* Advances in Neural Information Processing Systems (NeurIPS), 32. (TuRBO)
5. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* Proceedings of the 25th ACM SIGKDD, 2623–2631. (sampler TPE)
6. Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B., Wilson, A. G., & Bakshy, E. (2020). *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.* Advances in Neural Information Processing Systems (NeurIPS), 33. (usado no TuRBO)
7. Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* Proceedings of ICLR 2015.
8. Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems (NeurIPS), 32.
9. Virtanen, P. et al. (2020). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.* Nature Methods, 17, 261–272.

