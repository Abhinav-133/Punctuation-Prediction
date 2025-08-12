# Punctuation Prediction with BiLSTM (+ Transfer Learning)

This repository contains my end‑to‑end pipeline for **punctuation restoration** framed as a **sequence labeling** task. I trained two variants:

- **Baseline:** BiLSTM tagger with per‑timestep attention (random‑initialized embeddings)
- **Transfer Learning (TL):** Same architecture, but the embedding layer is **initialized from GloVe 6B.100d** and fine‑tuned

The model predicts one of the following labels **per token** (interpreted as the mark that follows that token):

```
O, COMMA, PERIOD, SEMICOLON
```
> In some runs I also experimented with `EXCLAMATIONMARK` and `QUESTIONMARK`, but they are extremely rare; for stable results I usually **drop or remap** them during preprocessing.

---

## ✨ Key features

- Clean **preprocessing** that converts inline punctuation into per‑token labels (labels attach to the **preceding** word)
- **Overlapping windows** (sequence length 50, stride ≈ 16) for long documents
- Compact **BiLSTM (2×128/dir)** + light **per‑step attention**
- **Class‑weighted Cross‑Entropy** (or **Focal Loss**) to handle the dominant `O` class
- **Transfer learning** option via pretrained **GloVe 6B.100d**
- Reproducible training with **best‑by‑val‑F1** checkpointing
- **Inference** that reconstructs readable text with small cleanup rules (no double punctuation, sensible endings)

---

## 🧱 Project structure (what I use)

```
punct_bilstm_a2z/
├─ data/
│  ├─ train.txt           # my training text with inline labels (e.g., ,COMMA .PERIOD ;SEMICOLON)
│  ├─ test.txt            # (optional) raw text to punctuate
│  └─ processed/          # created by preprocessing (train/val/test .jsonl)
├─ embeddings/
│  └─ glove.6B.100d.txt   # only for the TL model
├─ runs/                  # training outputs (checkpoints, vocabs, TensorBoard logs)
├─ results/               # saved evaluation reports / plots (optional)
├─ src/
│  ├─ preprocess.py       # CLI: build jsonl data files
│  ├─ train.py            # CLI: train baseline or TL
│  ├─ eval.py             # CLI: evaluate on val/test
│  ├─ infer.py            # CLI: punctuate text (stdin/file)
│  └─ ... (dataset.py, model.py, vocab.py, utils, etc.)
└─ tools/
   └─ plot_curves.py      # optional: overlay TB curves (baseline vs TL)
```

> Note: My `.gitignore` intentionally ignores **model files**, **large outputs**, and **all `*.txt`** (except things like `requirements.txt`).

---

## 🛠️ Setup

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install --upgrade pip
# If requirements.txt exists:
pip install -r requirements.txt
# Otherwise minimal deps:
pip install torch numpy scikit-learn tqdm matplotlib tensorboard
```

---

## 📄 Data

- Put your training text at: `data/train.txt`  
  (inline labels like `,COMMA` `.`**`PERIOD`** `;`**`SEMICOLON`** etc.)  
- (Optional) Put raw text to punctuate at: `data/test.txt`

If you need to **drop** `EXCLAMATIONMARK` and `QUESTIONMARK`, you can remove them from `train.txt` or (if supported by your `preprocess.py`) pass a flag (see below).

---

## 🔁 Preprocessing (creates `data/processed/*.jsonl`)

Basic split (train/val):
```bash
python -m src.preprocess \
  --input data/train.txt \
  --outdir data/processed \
  --val_ratio 0.10
```

Include a test file (when supported by your script):
```bash
python -m src.preprocess \
  --input data/train.txt \
  --test data/test.txt \
  --outdir data/processed \
  --val_ratio 0.10
```

Drop or remap ultra‑rare labels (if flags exist in your `preprocess.py`):
```bash
# drop ! and ?
python -m src.preprocess --input data/train.txt --outdir data/processed \
  --val_ratio 0.10 --drop_labels EXCLAMATIONMARK QUESTIONMARK

# or remap them to PERIOD
python -m src.preprocess --input data/train.txt --outdir data/processed \
  --val_ratio 0.10 --remap_labels EXCLAMATIONMARK:PERIOD QUESTIONMARK:PERIOD
```

Verify:
```bash
ls -l data/processed
# expect: train.jsonl, val.jsonl (and test.jsonl if provided)
```

---

## 🚂 Training

### A) Baseline BiLSTM
```bash
python -m src.train \
  --data_dir data/processed \
  --epochs 12 \
  --class_weights balanced
# (optional) --batch_size 32 --lr 1e-3 --dropout 0.3
```
Artifacts:
- Logs at `runs/bilstm/` (TensorBoard)
- Best checkpoint & vocabs: `runs/bilstm/best.pt`, `word_vocab.json`, `label_vocab.json`

### B) BiLSTM + Transfer Learning (GloVe)

1) Place pretrained vectors at: `embeddings/glove.6B.100d.txt`  
2) Train:
```bash
python -m src.train \
  --data_dir data/processed \
  --emb_path embeddings/glove.6B.100d.txt \
  --epochs 12 \
  --class_weights balanced
# (optional) lower LR on embeddings if supported: --emb_lr 5e-4 --lr 1e-3
```

> Tip: Use a separate run name/folder if your trainer supports it (e.g., `--run_name bilstm_tl`) to keep logs distinct.

---

## 🧪 Evaluation

```bash
python -m src.eval \
  --data_dir data/processed \
  --ckpt runs/bilstm/best.pt \
  --split val
# or: --split test
```
Outputs:
- Per‑class **Precision/Recall/F1**
- **Macro** & **Weighted** F1
- (Optional) Confusion matrix + classification report files in `results/`

I also report **Punctuation‑only Macro F1** over `{COMMA, PERIOD, SEMICOLON}` so the huge `O` class doesn’t dominate.

---

## ✍️ Inference (punctuate new text)

Single line (stdin):
```bash
echo "c is a compiled language it gives control to the programmer" | \
python -m src.infer --ckpt runs/bilstm/best.pt
```

Whole file → save to output:
```bash
mkdir -p outputs
python -m src.infer --ckpt runs/bilstm/best.pt < data/test.txt > outputs/test_punctuated.txt
```

Line‑aligned “before vs after” (great for examples/CSV):
```bash
: > outputs/test_punctuated_lines.txt
while IFS= read -r line; do
  printf "%s\n" "$line" | python -m src.infer --ckpt runs/bilstm/best.pt
done < data/test.txt > outputs/test_punctuated_lines.txt

# Optional: 2‑column CSV (Original, ModelOutput)
paste -d',' \
  <(sed 's/,/;/g' data/test.txt) \
  <(sed 's/,/;/g' outputs/test_punctuated_lines.txt) \
  > outputs/samples_before_after.csv
```

> If a command seems to hang, remember `infer.py` **waits for stdin**. Always pipe (`echo ... |`) or redirect (`< file`).

---

## 📊 Typical hyperparameters I use

- **Sequence length:** 50 (stride ≈ 16)
- **Embedding size:** 100 (random for baseline, **GloVe 6B.100d** for TL)
- **BiLSTM:** 2 layers, hidden **128/dir** (→ 256 per token)
- **Attention:** per‑timestep (Linear 256→1, softmax over time)
- **Dropout:** 0.3
- **Optimizer:** Adam (lr 1e‑3)
- **Loss:** Class‑weighted Cross‑Entropy (PAD ignored). Focal Loss is an option.
- **Scheduler:** ReduceLROnPlateau on val Macro‑F1 (factor 0.5, patience 3)
- **Grad clip:** 1.0
- **Epochs:** 8–15; pick best by **val Macro‑F1**

---

## 📈 Example results (Validation, TL run I trained)

```
Accuracy: 0.9099
Macro F1 (all 6 classes): 0.5822    • Weighted F1: 0.9275
Punctuation‑only Macro F1 (COMMA, PERIOD, SEMICOLON): 0.7605

Per‑class F1:
  O 0.9513 | COMMA 0.4357 | PERIOD 0.9271 | SEMICOLON 0.9186 | ! 0.1176 | ? 0.1429
```
Interpretation: PERIOD/SEMICOLON are strong; COMMA has high recall but low precision (model tends to over‑insert commas).

---

## 🧪 Repro & troubleshooting

- Fix Python/NumPy/Torch **seeds**.
- Keep length/stride/batch/LR/dropout **identical** across baseline vs TL to compare fairly.
- If you see `val.jsonl not found` → run **preprocess** first; check `--outdir` and `--data_dir` match.
- If you see `ModuleNotFoundError: src.model` → run **from project root** using `python -m src.train`.
- If eval can’t find vocabs → make sure you point `--ckpt` to the **right** `runs/...` folder.
- If baseline and TL logs mix → use different run names/folders (e.g., `runs/baseline_bilstm` vs `runs/bilstm_tl`).

---

## 📜 License
MIT (or your choice).

## 🙌 Acknowledgments
Pretrained word vectors from **GloVe (6B.100d)**.
