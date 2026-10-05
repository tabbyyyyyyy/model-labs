# Tabby 🐱 — a tiny GPT you can train on your laptop

A minimal, readable example of building and training a small language model
(a character-level, decoder-only transformer) from scratch with PyTorch.

```
tabby/
├── model.py        # TabbyGPT: embeddings → transformer blocks → LM head
├── tokenizer.py    # character-level tokenizer
├── train.py        # training loop (AdamW + OneCycle LR, train/val loss)
├── sample.py       # generate text from a checkpoint
├── data/input.txt  # a short story about Tabby the cat (training data)
└── requirements.txt
```

## Quick start (Windows PowerShell)

```powershell
cd D:\projects\tabby
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python train.py                       # ~1-2 min on CPU, ~0.8M params
python sample.py --prompt "Tabby " --length 300
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

## Train on your own text

Drop any UTF-8 `.txt` file in and point at it. More text = better model;
a few hundred KB to a few MB works well.

```powershell
python train.py --data data\my_text.txt --steps 3000
```

Make it bigger or smaller:

| flag           | default | what it does                     |
|----------------|---------|----------------------------------|
| `--n-layer`    | 4       | number of transformer blocks     |
| `--n-head`     | 4       | attention heads per block        |
| `--n-embd`     | 128     | hidden size (must divide by heads) |
| `--block-size` | 64      | context length in characters     |
| `--steps`      | 1000    | training steps                   |
| `--lr`         | 3e-3    | peak learning rate               |

## Notes

- The bundled story is tiny (~3 KB), so the model will mostly memorize it —
  watch `val` loss rise while `train` loss falls. That's overfitting, and a
  good lesson in why real models need lots of data.
- Uses a GPU automatically if `torch.cuda.is_available()`.
- Checkpoints are saved to `checkpoints/tabby.pt` plus `checkpoints/vocab.json`.
