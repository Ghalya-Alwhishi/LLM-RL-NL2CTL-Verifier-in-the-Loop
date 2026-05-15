# Beyond String Matching in LLM-Based NL-to-CTL Translation

This repository contains the reproducibility package for the paper:

**Beyond String Matching in LLM-Based NL-to-CTL Translation: Verifier-in-the-Loop Reinforcement Learning and Evaluation**

The project studies the translation of natural-language (NL) requirements into Computation Tree Logic (CTL) formulas that are directly usable by the NuSMV model checker. The framework combines LLM-based translation, NuSMV-guided syntax evaluation, verifier-grounded semantic evaluation, reinforcement-learning-based refinement stages, and end-to-end NL→CTL→NuSMV tools.

## Overview

The repository supports the main experimental workflow reported in the paper:

1. construction and validation of a NuSMV-compatible NL→CTL dataset;
2. supervised fine-tuning of T5 and Qwen3 models for NL→CTL translation;
3. NuSMV-guided syntax evaluation and refinement;
4. verifier-grounded semantic evaluation using truth-set agreement over Kripke/SMV models;
5. within-model and cross-model comparisons between T5 and Qwen3;
6. deployment of interactive Streamlit tools for NL→CTL translation and optional NuSMV verification;
7. cybersecurity case-study evaluation.

## Repository Structure

```text
.
├── README.md
├── README_reproduction.md
├── requirements.txt
│
├── datasets/
│   ├── Cleaned_CTL_Dataset_nusmvOK.csv
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
│
├── models_or_checkpoints/
│   └── checkpoint_instructions.md
│
├── results/
│   ├── t5/
│   ├── qwen3/
│   
│
├
├── case_study/
│   ├── cybersecurity_model.smv
│   ├── cybersecurity_requirements.csv
│   └── case_study_outputs.csv
│
├── tools/
│   ├── t5_streamlit_app.py
│   └── qwen_streamlit_app.py
│
└── scripts/
    ├── syntax_eval_nusmv.py
    ├── semantic_truthset_eval.py
    ├── train_sft.py
    