# A small GPT from scratch

A character-level GPT in PyTorch. I wrote the masked multi-head self-attention, the MLP and
the transformer blocks myself (`model.py`); no `nn.Transformer` or `nn.MultiheadAttention`.
Trained on a laptop CPU in about 20 minutes.

## Training text

*Indian Fairy Tales* by Joseph Jacobs (1892), Project Gutenberg ebook #7128
(https://www.gutenberg.org/ebooks/7128), which is in the public domain. `python data/prepare.py` downloads it and strips the Gutenberg header and
licence footer (395,000 characters). The text file itself is not committed.

## Run

```
pip install torch                      # the CPU build is enough
python data/prepare.py
python train.py --steps 1500           # saves out/ckpt.pt every 250 steps
python train.py --steps 2000 --resume  # carries on from the checkpoint
python generate.py "Once upon a time" --temperature 0.8 --top_p 0.9
python -m unittest discover -s tests
```

Model files (`out/`, `*.pt`) are not committed. `train_log.txt` is the loss log, and
`results/` has the same log as JSON plus the samples.

## Model

| setting | value |
|---------|-------|
| vocabulary | 97 characters |
| context | 128 characters |
| layers / heads / embedding | 4 / 4 / 128 |
| parameters | 834,560 |
| dropout | 0.1 |

- `CausalSelfAttention`: one linear layer makes queries, keys and values; they are split into
  4 heads; scores are `q·k / sqrt(head_dim)`; a lower-triangular mask sets future positions
  to `-inf` before the softmax, so position t only sees positions up to t; heads are joined
  and projected back.
- `Block`: pre-norm, `x + attention(layernorm(x))` then `x + mlp(layernorm(x))`.
- `GPT`: token embedding + learned position embedding, 4 blocks, a final layer norm and a
  linear head to the vocabulary.

## Training

- The first 90% of the book is for training and the last 10% for validation; validation text
  is never trained on. Both losses are estimated every 250 steps (20 batches each).
- AdamW, learning rate 3e-4, batch 32, gradient clipping at 1.0.
- Every evaluation saves `out/ckpt.pt` with the model, the optimiser state, the step, the
  config and the vocabulary. `--resume` loads all of it and continues from that step.

`train_log.txt` (the second run resumed at step 1500):

```
step 0:    train 4.794  val 4.794
step 500:  train 2.388  val 2.802
step 1000: train 2.123  val 2.592
step 1500: train 1.906  val 2.413
resumed from step 1500
step 1750: train 1.825  val 2.355
step 2000: train 1.745  val 2.279
```

`results/train_log.jsonl` has the same numbers as JSON.

## Sampling

`sample_next` divides the logits by the temperature, sorts the probabilities, keeps the
smallest set of tokens whose total reaches `top_p` (always at least one), renormalises and
samples. Temperature 0 picks the most likely token. Tests check greedy decoding, that top-p
drops unlikely tokens, and that `top_p = 1` keeps every token.

`results/samples.txt`, all starting from "Once upon a time" with top_p 0.9:

- **0.5:** "Once upon a time the princess of his and strack, and the prince the with prince a a sone
  the princess of the pated the for a good..."
- **0.8:** "Once upon a time marry and mor ment of it is a con and the sarrie the mestrother..."
- **1.2:** "Once upon a time marry muld. On Eve famers, Lassie daratess daugar hed."

## What the model gets wrong, and why

- **It writes English-looking nonsense.** It has learned spelling patterns, common short words
  ("the", "and", "to"), quotation marks, capitals after full stops and fairy-tale words
  ("princess", "King"), but not meaning or grammar. With 835k parameters, 2,000 steps and one
  short book, that is what to expect: a validation loss of 2.28 nats per character is far from
  fluent text (good character models reach about 1.0 to 1.3).
- **Low temperature repeats itself** ("the prince ... the princess ... the princess"), because
  it keeps choosing the few safest characters. **High temperature invents words**
  ("daratess", "praids") because unlikely characters get picked more often.
- **It is overfitting a little.** Train loss (1.75) is below validation loss (2.28) and the gap
  grew from 0.41 at step 500 to 0.53 at step 2000. The validation text is the end of the book,
  with stories and names the model never saw, which makes it harder.
- **It has no memory beyond 128 characters**, so it cannot keep a story or a character going.
- **Character-level tokens are slow:** every word costs several predictions. A BPE tokenizer
  would let the same model see much more text per step.

More data, a bigger model and more steps (for example on Colab's free GPU) would help most.
