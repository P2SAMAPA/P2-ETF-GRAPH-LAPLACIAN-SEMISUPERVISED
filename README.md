# Graph Laplacian Semi‑Supervised Learning for ETFs

Applies label propagation on the ETF correlation graph. Initial labels are based on the sign of the last day's return (positive/negative). The graph Laplacian regularises the label assignment, producing a probability of being in the positive class for every ETF. This signal captures momentum‑connectedness.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- k‑NN graph from correlation distance (1 - |correlation|)
- Label propagation via solving (I + α L) f = y
- Score = probability of positive class after propagation
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-graph-laplacian-semisupervised-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF is central to the cluster of recently positive ETFs → expected to continue upward.
- Low score → ETF is disconnected from positive momentum.

## Requirements

See `requirements.txt`.
