## BERT / RoBERTa Classifier (Simple Explanation — Like I Explain to Someone)

So basically, in our project we use RoBERTa as one of the main models for fake news detection.

Before using it, there is an important concept called **fine-tuning**.

---

### What is Fine-Tuning?

RoBERTa is already a pretrained model.
It has learned:

* Grammar
* Word meanings
* Context
* General knowledge

This training is done on a very large dataset (around 160GB of text).

Now, instead of training a model from scratch, we do **fine-tuning**.

That means:

* We take the already trained RoBERTa
* Add a small classification layer on top
* Train it on our dataset (LIAR)
* So it adapts to our specific task (fake vs real news)

---

### Simple Analogy

Training from scratch:
Teaching someone how to read + then teaching them a subject

Fine-tuning:
Taking someone who already knows how to read and just teaching them the subject

---

### What Changes Inside the Model?

When we fine-tune:

* The model weights update slightly
* The classification layer is trained properly
* The model adjusts to our dataset

But we don’t change everything aggressively.

---

### Why Not Update Everything Fully?

If we update all weights too much on a small dataset (10,000 samples):

* The model forgets its original knowledge
* It becomes too focused on our dataset
* This is called **overfitting**

In research terms, this is called:
👉 **Catastrophic Forgetting**

---

### Solution

We use a very small learning rate (2e-5)

This ensures:

* Old knowledge is preserved
* New knowledge is learned slowly

---

### Classification Head (Important Part)

RoBERTa does not directly give output like FAKE or REAL.

It gives numbers (vectors).

So we add a classification layer:

Input Text
→ RoBERTa
→ Vector
→ Linear Layer
→ Softmax
→ Final Prediction

---

### What is the [CLS] Token?

When we give input to RoBERTa, it automatically adds special tokens.

Example:

Input:
vaccines cause autism

Actual input:
[CLS] vaccines cause autism [SEP]

---

### Why [CLS] is Important?

The [CLS] token is special.

It represents:
👉 The overall meaning of the entire sentence

After processing:

* Every word has its own vector
* But [CLS] stores the full sentence understanding

---

### Why Do We Use [CLS]?

Because:

* It already summarizes the whole sentence
* It is designed for classification tasks

So we take:
👉 Only the [CLS] vector

---

### Dimensions

Each token becomes a vector of:
👉 768 numbers

So:
[CLS] → 768-dimensional vector

---

### Final Flow

Input:
"[CLAIM] vaccines cause autism"

→ RoBERTa processes text
→ [CLS] vector (768 values)
→ Linear layer converts 768 → 2
→ Softmax converts into probabilities

Example output:
[0.87, 0.13]
FAKE   REAL

Prediction:
FAKE (higher value)

---

### Why Not Average All Words?

If we average all words:

* Every word gets equal importance
* Even useless words like “the”

But:
👉 [CLS] already understands importance of words

So it is better.

---

## Final Understanding

* RoBERTa understands language deeply
* Fine-tuning adapts it to our task
* Small learning rate prevents overfitting
* [CLS] token gives full sentence meaning
* Classification layer gives final output

So this model helps us detect fake news based on language patterns and context.
