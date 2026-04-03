# Accountable Medical Triage

A live demo showing how conformal prediction turns an overconfident LLM into a safer triage system.

GPT-4o-mini classifies patient symptoms into 4 triage levels. It's 94% confident even when it's wrong. Conformal prediction wraps the model with calibrated uncertainty — building prediction sets that include all plausible triage levels instead of betting on a single answer.

**[Live Demo →](https://conformal-triage-demo.streamlit.app/)**

## What it does

- Extracts real logprobs from GPT-4o-mini at the classification token
- Applies conformal temperature scaling (T=0.5) to optimize the probability distribution
- Uses split conformal prediction (threshold method) to build prediction sets with coverage guarantees
- On 320 test cases: 87.8% coverage, 31 patients "saved" where CP caught errors the model missed

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
streamlit run app.py
```

The demo works without an API key — precomputed examples and the full test gallery load from `data/precomputed.pkl`. Live predictions on custom input require the key.

## Project structure

```
app.py                  # Streamlit demo (single file)
precompute.py           # Pipeline: inference → calibration → prediction sets
config.yaml             # Model and CP parameters
src/
  conformal/engine.py   # Split conformal prediction, threshold scoring
  conformal/metrics.py  # Coverage, set size, accuracy metrics
  conformal/weighted.py # Tibshirani weighted CP (experimental)
  model/inference.py    # GPT-4o-mini logprob extraction
  ood/embeddings.py     # OOD detection via sentence embeddings
data/
  precomputed.pkl       # Precomputed results for 800 cases
  synthetic_triage_*.json  # Synthetic patient data (ESI 1-5)
docs/
  DESIGN_SYSTEM.md      # UI/UX design system (Maison Noire)
```

## The method

Split conformal prediction with threshold scoring. A held-out calibration set (480 cases) determines the probability cutoff: any triage level above 0.41% enters the prediction set. That threshold is tiny because the model is trained to be overconfident — it shoves nearly all probability mass onto one class. Even 0.41% is a meaningful signal.

Coverage guarantee: the true triage level is in the prediction set at least 90% of the time.

## References

- Vovk, Gammerman & Shafer. *Algorithmic Learning in a Random World* (2005) — foundational conformal prediction framework
- Angelopoulos & Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification* (arXiv:2107.07511, 2022) — accessible tutorial covering split conformal prediction and threshold scoring
- Tibshirani. *Conformal Prediction* — Advanced Topics in Statistical Learning, UC Berkeley, Spring 2023 — lecture notes covering split conformal, coverage guarantees, and practical implementation
- Candès, Lei & Wasserman. *Predictive Inference with the Jackknife+* (Annals of Statistics, 2019) — distribution-free coverage guarantees
- Tibshirani, Barber, Candès & Ramdas. *Conformal Prediction Under Covariate Shift* (NeurIPS 2019) — weighted conformal prediction for distribution shift
- Cherian, Gibbs & Candès. *Conformal Prediction Under Covariate Shift* (NeurIPS 2024) — covariate shift extensions used as academic foundation for this project
- ESI Triage Research Team. *Emergency Severity Index v5* — the clinical triage framework mapped to our 4-class system

## Built by

Lotta-Lorette Kalmaru · 2026
