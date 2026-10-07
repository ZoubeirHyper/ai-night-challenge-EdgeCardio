# EdgeCardio

Multi-label 12-lead ECG classifier, quantized to int8 for edge deployment on a Raspberry Pi.

Built at the **AI Night Challenge** hackathon (team of 5) and later reworked with a patient-safe evaluation split.

> **Disclaimer:** research and learning prototype. Not a medical device and not for clinical use.

---

## What it does

Given a 10-second, 12-lead ECG, the model predicts which of five diagnostic superclasses apply. A record can belong to several classes at once (multi-label).

| Class | Meaning |
|---|---|
| NORM | Normal ECG |
| MI | Myocardial infarction |
| STTC | ST/T change |
| CD | Conduction disturbance |
| HYP | Hypertrophy |

## Dataset

[PTB-XL](https://physionet.org/content/ptb-xl/) (PhysioNet): about 21,800 clinical 12-lead ECG records of 10 seconds each. Labels are the five diagnostic superclasses derived from the SCP-ECG statements in `scp_statements.csv`.

The dataset is not included in this repo. Download it from PhysioNet and point `preprocess.py` to it.

## Pipeline

```
PTB-XL  ->  preprocess.py  ->  train.py  ->  newmodel.onnx  ->  quantize.py  ->  newmodel_int8.onnx
                                                                                      |
                                                                          infer_onnx.py / model_tester.py
```

- **Model:** small 1D CNN (4 convolutional blocks with batch norm, global average pooling, 2-layer classifier head), trained with `BCEWithLogitsLoss`.
- **Training:** Adam, learning-rate scheduling on validation macro-AUC, early stopping.
- **Export:** PyTorch to ONNX, then dynamic int8 weight quantization with ONNX Runtime.
- **Metric:** macro-AUC (the competition metric), plus per-class AUC and F1.

## Results

> Fill this table with the numbers from your own v2 run. Do not leave placeholders in the published version.

Evaluation uses the official PTB-XL split: folds 1-8 train, fold 9 validation, fold 10 test.

| Model | Macro-AUC | NORM | MI | STTC | CD | HYP | Size | Latency |
|---|---|---|---|---|---|---|---|---|
| fp32 (ONNX) | TBD | TBD | TBD | TBD | TBD | TBD | TBD MB | TBD ms |
| int8 (ONNX) | TBD | TBD | TBD | TBD | TBD | TBD | TBD MB | TBD ms |

Latency measured on: TBD (device, CPU threads, input length).

## Project structure

```
EdgeCardio/
  models/
    newmodel.pt           # trained PyTorch weights
    newmodel.onnx         # fp32 ONNX export
    newmodel_int8.onnx    # int8-quantized model for edge inference
  scripts/
    preprocess.py         # PTB-XL -> .npy arrays (official fold split)
    model.py              # CNN architecture
    train.py              # training, validation, ONNX export
    quantize.py           # int8 quantization + latency benchmark
    infer_onnx.py         # inference demo with confidence per class
    model_tester.py       # test-set evaluation (AUC, per-class report)
  output.mp4              # demo video
  README.md
```

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess PTB-XL
python scripts/preprocess.py --data_dir /path/to/ptb-xl --out_dir data

# 3. Train and export to ONNX
python scripts/train.py

# 4. Quantize to int8
python scripts/quantize.py

# 5. Evaluate on the test fold
python scripts/model_tester.py

# 6. Run the inference demo
python scripts/infer_onnx.py
```

## Version history

- **v1 (hackathon):** the first working pipeline, built during the event. It used a random train/validation/test split, which can place records from the same patient in both train and test, so its scores are likely optimistic. Tagged as `hackathon-v1`.
- **v2:** switched to the official PTB-XL `strat_fold` split (no patient overlap between folds), unified the input length across all scripts, moved normalization into preprocessing, and added fp32 vs int8 comparison.

## Limitations

- Trained and evaluated on a single dataset (PTB-XL), so performance on other hospitals, devices or populations is unknown.
- Label noise: PTB-XL labels come from automated and human annotations of varying certainty.
- Dynamic int8 quantization can change per-class performance slightly; see the results table for the measured difference.
- Fixed decision thresholds were tuned on the validation fold only.
- Not validated clinically.

## Team

Built at the AI Night Challenge by a team of 5:
- TBD (add names and GitHub links)

## Acknowledgements

- [PTB-XL dataset](https://physionet.org/content/ptb-xl/), Wagner et al., PhysioNet.
- ONNX Runtime for quantization and inference.
