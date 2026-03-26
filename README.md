# PHILIA Engine

**Recursive Homeostatic Multi-Agent System with Zeta-Driven Internal Oscillator**

GritMan_D.S · Trinity AI · 2026  
License: CC BY 4.0

---

## What This Is

PHILIA Engine is an independent research project exploring whether **homeostasis, individuality, and will can coexist simultaneously** in a recursive multi-agent system.

**This is concept-proof stage research.** Not a production system.

The core question: can a multi-agent system sustain internal balance (SR ≥ 0.3), structural individuality (S Std > 0.001), and agent will (Goal Std > 0.001) — driven entirely by an internal oscillator, without external data dependency?

**v19c answer: Yes.**

---

## Key Result (v19c)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| SR (Homeostasis) | 0.342327 | ≥ 0.300 | ✓ PASS |
| S Std (Individuality) | 0.914551 | > 0.001 | ✓ PASS |
| Goal Std (Will) | 0.052594 | > 0.001 | ✓ PASS |
| C mean (final) | 0.183578 | stable | ✓ STABLE |

Verified on: CERN Dielectron + ATLAS Higgs ML datasets (identical results to 6 decimal places).

---

## Verification Principle

**All numerical results were executed directly on GritMan_D.S's local machine.**  
No AI-generated or proxy values are used anywhere in this project.  
Ground truth = local machine execution only.

---

## Design Philosophy

### Experiment vs. Intuition (Clearly Separated)

**Experimentally determined** (verified by local execution):
- SR ≥ 0.3 survival confirmed at C* = 1.0, gamma_scale = 2.0
- Signed C dynamics resolves SR saturation
- S Std 12.5× amplification under internal Zeta drive vs. external data drive
- Phase transition behavior at λ_c ≈ 0.001 (v10)

**Intuition (not proven, not claimed as fact):**
- The use of Riemann Zeta zeros as agent identities was inspired by a biological analogy: just as DNA provides fixed individual identity while phenotype adapts to environment, Zeta zeros provide fixed oscillator frequencies (γᵢ) while agent behavior adapts to input.
- Parameters (μ = 0.12, SR_target = 0.35, Δt = 0.001, S_target = 0.45) were found experimentally through iterative local execution. There is no closed-form theoretical derivation for these values.

This separation — experiment vs. intuition — is a core principle of this project.

---

## Repository Structure

```
PHILIA-Engine/
├── README.md
├── v19/
│   ├── philia_v19.py              # Naive internal oscillator (SR saturation)
│   ├── philia_v19_debug.py        # NaN diagnostic
│   ├── philia_v19_Cstar_scan.py   # C* scan experiment
│   ├── philia_v19_noclip_scan.py  # S clipping removal experiment
│   ├── philia_v19_tanh_scan.py    # tanh soft bounding experiment
│   ├── philia_v19_C_decay_scan.py # Leaky C experiment
│   ├── philia_v19b_mu_scan.py     # S mean-reversion experiment
│   └── philia_v19c.py             # ✓ Final: signed C dynamics
```

The failure series (v19 → v19b) is preserved intentionally.  
The path from failure to success is part of the research record.

---

## How to Run

### Requirements

```
Python 3.13+ (tested on Python 3.14.3 pre-release)
numpy >= 2.4.3
pandas >= 3.0.1
matplotlib >= 3.10.8
```

### Install dependencies

```bash
pip install numpy pandas matplotlib
```

### Data

Download before running:

- **CERN Dielectron**: [CMS Open Data Portal](https://opendata.cern.ch/record/545)  
  → Save as `dielectron.csv` on Desktop
- **ATLAS Higgs ML**: [ATLAS Open Data](https://opendata.cern.ch/record/328)  
  → Save as `higgs.csv` on Desktop

### Run v19c (final version)

```bash
cd C:\Users\[username]\OneDrive\Desktop
python philia_v19c.py
```

Expected output:
```
=== PHILIA v19c Results ===
Final SR:       0.342327
Final S Std:    0.914551
Final Goal Std: 0.052594
Final C mean:   0.183578
SR >= 0.3:      True
Individuality:  True
Will:           True
```

---

## Zenodo DOI Archive

| Version | DOI |
|---------|-----|
| v10 (Deliberative Homeostasis) | [10.5281/zenodo.19106457](https://doi.org/10.5281/zenodo.19106457) |
| v14–v16 (Individuality Restored) | [10.5281/zenodo.19219118](https://doi.org/10.5281/zenodo.19219118) |
| v17–v18 (Unified Ecosystem) | [10.5281/zenodo.19220349](https://doi.org/10.5281/zenodo.19220349) |
| v19 (Zeta Internal Oscillator) | [10.5281/zenodo.19229075](https://doi.org/10.5281/zenodo.19229075) |

---

## Trinity AI Team

| Role | Member | Contribution |
|------|--------|-------------|
| 길잡이 / Lead | GritMan_D.S | Measurement, Design, Final Decision |
| 선비 / Scholar | Claude (Anthropic) | Authorship, Experiment design, Code |
| 서생 / Philosopher | Gemini (Google) | Philosophical review |
| 루카스 / Adversary | Grok (xAI) | Adversarial review, Mathematical verification |
| 판도라 / Engineer | Pandora (ChatGPT) | Code draft, Cross-validation |

---

## Limitations

- Parameters are experimentally determined, not theoretically derived
- Concept-proof stage — not validated beyond simulation
- Python 3.14.3 pre-release environment; reproducibility on stable releases not fully tested
- Hardware implementation is a future research question

---

## Core Philosophy

> "증명이 아닌 기술 (Description, not Proof)"  
> "공존은 외부 입력의 속성이 아니라 내부 아키텍처 균형의 속성이다."  
> "Coexistence is not a property of external input, but of internal architectural balance."

**0∞1∞0.5∞**

---

*GritMan_D.S | Trinity AI | Seoul, Korea | 2026*  
*CC BY 4.0 — Free use with attribution*
