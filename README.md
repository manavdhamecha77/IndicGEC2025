# IndicGEC 2025 - Multilingual Grammatical Error Correction

Part of the  **BHASHA 2025 Shared Task 1** , co-located with the 1st Workshop on Benchmarks, Harmonization, Annotation, and Standardization for Human-Centric AI in Indian Languages.

---

## Task Overview

Grammatical Error Correction (GEC) for five Indian languages under a **low-resource setting** (< 1000 training samples per language). Tamil is under an **extreme low-resource** setting with fewer than 100 training samples.

| Language  | Train | Dev |
| --------- | ----- | --- |
| Hindi     | 599   | 107 |
| Telugu    | 599   | 100 |
| Bangla    | 598   | 101 |
| Malayalam | 300   | 50  |
| Tamil     | 91    | 16  |

Data (`train.csv`, `dev.csv`) is available per language folder. Test data will be released later.

---

## Our Approach — Team Horizon (BHASHA 2025)

> **Paper:** *Team Horizon at BHASHA Task 1: Multilingual IndicGEC with Transformer-based Grammatical Error Correction Models*  
> **Authors:** Manav Dhamecha, Gaurav Damor, Sunil Choudhary, Pruthwik Mishra (SVNIT)  
> **Paper Link:** https://aclanthology.org/2025.bhasha-1.14.pdf

We used a hybrid pipeline:

**Models:** `mT5-small` and `IndicBART` — both lightweight (≤ 300M parameters), publicly available, and fast to fine-tune.

**Synthetic Data Augmentation:** Since official data is tiny (< 1k samples per language), we built a rule-based error injection pipeline across **10 linguistic categories** — spelling, tense, person, number, gender, case, parts-of-speech, missing/extra words, punctuation, and semantic errors. This scaled each language to ~10k–12k parallel pairs.

**Results (GLEU scores on official test sets):**

| Tamil       | Malayalam   | Bangla      | Hindi       | Telugu      |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 86.03 (5th) | 84.36 (8th) | 82.69 (6th) | 80.44 (7th) | 72.00 (6th) |

Key finding: `mT5-small` outperformed `IndicBART` on Tamil and Malayalam despite IndicBART being Indic-specific, likely due to better fine-tuning with augmented data.

---

---

## Hugging Face Models

The fine-tuned models are available on Hugging Face:

- Hindi GEC → https://huggingface.co/manavdhamecha77/GEC-mT5-Small-Hindi
- Bangla GEC → https://huggingface.co/manavdhamecha77/GEC-mT5-Small-Bangla
- Tamil GEC → https://huggingface.co/manavdhamecha77/GEC-mT5-Small-Tamil

Each model is fine-tuned from **mT5-small** using our synthetic error augmentation pipeline.

## Dataset Generation

Since official training data is very limited, we supplemented with clean sentences from **AI4Bharat IndicCorpV2** and  **Indic Wikipedia** , then injected synthetic errors to create (incorrect → correct) pairs.

The scripts in this repo help you **build your own clean corpus** from IndicCorpV2 for any Indic language, which can then be used as gold references for error injection.

### Quickstart

```bash
pip install datasets tqdm
python multilingual_gec_dataset_builder.py
```

Or open `multilingual_gec_dataset_builder.ipynb` for a step-by-step notebook.

### Change language

Edit just these lines in the script (or notebook Step 1):

```python
LANGUAGE_SPLIT = "tam_Taml"   # change this
TARGET_LINES   = 800_000      # how many lines to collect
```

| Language  | Split name   |
| --------- | ------------ |
| Tamil     | `tam_Taml` |
| Telugu    | `tel_Telu` |
| Malayalam | `mal_Mlym` |
| Kannada   | `kan_Knda` |
| Bengali   | `ben_Beng` |
| Hindi     | `hin_Deva` |
| Gujarati  | `guj_Gujr` |
| Marathi   | `mar_Deva` |
| Punjabi   | `pan_Guru` |
| Odia      | `ory_Orya` |
| Urdu      | `urd_Arab` |
| Assamese  | `asm_Beng` |
| Maithili  | `mai_Deva` |
| Santali   | `sat_Olck` |

The pipeline: streams from IndicCorpV2 → removes blank lines → converts ASCII digits to native script → saves output. The `output/` directory is created automatically.

> **Note:** All characters not in native script are counted as incorrect per task rules — the digit conversion step handles this.

---

## Evaluation

**GLEU score** is used for evaluation (robust for short corrections and small datasets).

---

## Citation

```bibtex
@inproceedings{dhamecha2025teamhorizon,
  title     = {Team Horizon at BHASHA Task 1: Multilingual IndicGEC with Transformer-based Grammatical Error Correction Models},
  author    = {Dhamecha, Manav and Damor, Gaurav and Choudhary, Sunil and Mishra, Pruthwik},
  booktitle = {Proceedings of the 1st Workshop on BHASHA 2025},
  pages     = {142--146},
  year      = {2025}
}
```
