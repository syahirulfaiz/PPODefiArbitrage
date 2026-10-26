# Constrained DRL for Sequential Atomic Triangular Arbitrage (TNSE v2.10)

Reproducible code for the manuscript *"A Constrained Deep Reinforcement Learning
Framework for Atomic DeFi Triangular Arbitrage"* (IEEE TNSE resubmission).

## 1. What this project does

- Formulates **sequential atomic triangular arbitrage** as constrained cumulative
  after-gas profit maximization (constraints C1–C6; block-time constraint `H` = 12 s
  on the cumulative decision procedure).
- Trains a **two-head residual PPO** policy: one categorical head selects an
  **ordered triangular path**, one continuous head outputs a **first-swap input**
  (later swap amounts follow the CPMM).
- Compares **9 methods** (PPO, SAC, TD3, Greedy, Heuristic, Beam Search,
  Grid Search, Bellman-Ford, constrained NLP) on **23 one-factor settings × 10
  held-out seeds = 2,070 episodes** (230 per method).
- Uses a **V2-style CPMM simulator** over a fixed snapshot: **174 pools**,
  **154 canonical token symbols**, **114 candidate ordered triangular paths**
  (27 June 2026). Uniswap V3 concentrated liquidity is out of scope.

## 2. Repository layout

```
01.CODE/
├── train.py                            # Optuna HPO + final training (PPO/SAC/TD3)
├── driver.py                           # 2,070-episode evaluation of 9 methods
├── chart.py                            # renders the paper figures from the workbook
├── requirements.txt
├── slurm_train.example.sh              # HPC training job template (fill <PLACEHOLDER>s)
├── slurm_driver.example.sh             # HPC evaluation job template (fill <PLACEHOLDER>s)
├── DATASET/                            # pool / token-price / wallet snapshots + generators
├── models/                             # checkpoints, HPO trials, selection.json (generated)
├── results/                            # raw_runs.csv, summary CSVs, workbook, figures/ (generated)
└── logs/                               # Slurm stdout/stderr
```

## 3. Environment

```bash
module purge
module load miniconda/3
conda create -n ppr01 python=3.10 -y
conda activate ppr01
pip install -r requirements.txt
```

Reference environment used for the paper:
A Python 3.10 environment with the packages in requirements.txt
(PyTorch 2.12, Stable-Baselines3 2.9, Gymnasium 1.3, Optuna 4.5, SciPy 1.15,
NetworkX 3.4, pandas 2.3, matplotlib 3.10).

## 4. Dataset

`DATASET/` contains:

| File | Content |
|---|---|
| `uniswap_v2_and_sushiswap_v2_pools_data.2026-06-27.json` | **174 pools** (89 Uniswap V2 + 85 SushiSwap V2) |
| `token_prices_2026-06-27.json` | USD token prices (CoinGecko) |
| `wallet_balance_2026-06-27.<1000..10000>.json` | initial wallets split equally across DAI, ETH, USDT, USDC, WBTC |
| `*.mhtml` | raw Uniswap/SushiSwap pool-page snapshots |
| `uniswap_v2_and_sushiswap_v2_pools_data_generator.py` | regenerates the pool JSON |
| `token_price_generator.py` | regenerates the token-price JSON |

Gas costs are sampled by an **empirical bootstrap of paired gas-price/gas-usage
rows** taken from real Ethereum transactions (Etherscan). No KDE model is used.
The experiment is a controlled snapshot simulation, not a historical replay and
not an on-chain deployment .

## 5. Reproduce (two steps + figures)

```bash
# 1) HPO + final training for PPO / SAC / TD3
python train.py \
  --dataset-dir DATASET \
  --output-dir models \
  --trials 12 \
  --hpo-steps 100000 \
  --steps 1000000 \
  --jobs 3

# 2) Evaluation: 23 settings x 9 methods x seeds 40-49 = 2,070 episodes
python driver.py \
  --dataset-dir DATASET \
  --model-dir models \
  --output-dir results \
  --workbook PPR01_V209_EVALUATION_job9851603.xlsx

# 3) Figures
python chart.py --input results/PPR01_V209_EVALUATION_job9851603.xlsx
```

On an HPC cluster, create `logs/` first, then submit the provided templates after
replacing the `<PLACEHOLDER>` fields (`<PARTITION>`, `<ACCOUNT>`, `<PROJECT_ROOT>`, `<CONDA_ENV>`):

```bash
sbatch slurm_train.example.sh
sbatch slurm_driver.example.sh
```

## 6. Protocol (matches the manuscript)

- Training seed **20**; validation seeds **30–39**; held-out test seeds **40–49**.
- 12 Optuna trials × 100k steps; final training 1M steps; checkpoints every **250k**.
- Checkpoint selection: best mean cumulative after-gas profit on the 10 validation
  episodes (deterministic policy prediction); ties broken by lower mean complete
  decision time than by higher training timesteps.
- Constraints: C1 positive after-gas profit, C2 wallet balance, C3 directional
  reserve, C4 swap-leg continuity, C5 path closure, C6 cumulative decision time ≤ `H`.
- Metrics: simulated after-gas profit, ROI, simulated reward-to-risk ratio
  (mean ROI / std ROI), gas-cost share, arbitrage type (single-DEX / cross-DEX),
  and **complete decision time**.
- All methods share the simulator, initial state, gas-sampling seed and constraints,
  but keep their native decision procedures.

## 7. Outputs

- `results/raw_runs.csv` — one row per episode
- `results/overall_by_method.csv`, `summary_by_value_method.csv`, `config.csv`,
  `scenario_plan.csv`, `ppo_pairwise_counts.csv`
- `results/PPR01_V209_EVALUATION_job9851603.xlsx` — multi-sheet workbook
- `results/figures/*.pdf|png` — paper figures (Fig. 4–10) and `atlas_cells.csv`

## 8. Notes and scope

- This is a **V2-style constant-product** simulator. Concentrated liquidity, tick
  crossing, and active-liquidity dynamics of Uniswap V3 are out of scope.
- Paths are constructed from **canonical token symbols**, so continuity is
  symbol-level rather than contract-level.
- Reported returns characterize the implemented policies under one fixed snapshot.
  They do not establish executable on-chain profitability.

## 9. Citation

```bibtex
@article{faiz2026constrained,
  title   = {A Constrained Deep Reinforcement Learning Framework for Atomic DeFi Triangular Arbitrage},
  author  = {Faiz, Syahirul and Materwala, Huned and Svetinovic, Davor},
  journal = {IEEE Transactions on Network Science and Engineering (under review)},
  year    = {2026}
}
```

