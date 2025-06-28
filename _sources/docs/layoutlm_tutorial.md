
# 🧾 LayoutLM + SROIE Tutorial (Part 1)

> ✍️ **Goal**: Extract structured information from scanned receipts using LayoutLM and the SROIE dataset.

---

## 📌 Table of Contents

1. [Overview](#overview)
2. [Dataset: SROIE](#dataset-sroie)
3. [Introduction to BIO Tagging](#introduction-to-bio-tagging)
4. [Input Structure for LayoutLM](#input-structure-for-layoutlm)
5. [Common Pitfalls & Notes](#common-pitfalls--notes)
6. [Drill-Down Exercise: Tagging + Token Inputs](#drill-down-exercise-tagging--token-inputs)
7. [Next Steps](#next-steps)

---

## 🧠 Overview

LayoutLM is a transformer-based model designed to work with documents by incorporating:
- **Text**
- **Layout information (bounding boxes)**
- **(Optional) Visual features from the image**

It's ideal for tasks like **Named Entity Recognition (NER)** or **key-value extraction** in scanned documents such as receipts.

---

## 📁 Dataset: SROIE

SROIE (Scanned Receipt OCR and Information Extraction) contains:
- **Images** of receipts
- OCR-extracted **text files**
- Ground truth **entity annotations** for:
  - `COMPANY`
  - `DATE`
  - `ADDRESS`
  - `TOTAL`

### 📦 Structure (example):

```
sroie/
├── img/
│   └── receipt001.jpg
├── ocr/
│   └── receipt001.txt
└── labels/
    └── receipt001.txt
```

---

## 🔖 Introduction to BIO Tagging

BIO tagging is used to annotate tokens in sequence tasks like NER.

| Tag Prefix | Meaning                  |
|------------|--------------------------|
| B-         | Beginning of an entity   |
| I-         | Inside an entity         |
| O          | Outside any entity       |

### 🧾 Example

Sentence:
```
SuperMart  2024-12-01  Total:  $14.23
```

Target fields:
- COMPANY: `SuperMart`
- DATE: `2024-12-01`
- TOTAL: `$14.23`

| Token         | BIO Tag      |
|---------------|--------------|
| SuperMart     | B-COMPANY    |
| 2024-12-01    | B-DATE       |
| Total         | O            |
| :             | O            |
| $14.23        | B-TOTAL      |

---

## 🧩 Input Structure for LayoutLM

Each token is converted into multiple components before being passed to the model:

| Field             | Description                                 |
|------------------|---------------------------------------------|
| `input_ids`      | Token IDs from tokenizer                    |
| `bbox`           | Bounding box `[x0, y0, x1, y1]` (0–1000 scale) |
| `attention_mask` | 1 = attend to, 0 = padding                  |
| `labels`         | BIO tag index for each token (training only) |

### 📘 Example Input

```python
input_ids = [1234, 5678, 4321, 1001, 8765]

bbox = [
  [50, 100, 200, 150],
  [210, 100, 330, 150],
  [340, 100, 400, 150],
  [401, 100, 410, 150],
  [420, 100, 490, 150],
]

attention_mask = [1, 1, 1, 1, 1]
labels = [1, 3, 0, 0, 5]
```

---

## ⚠️ Common Pitfalls & Notes

| Area | Misunderstanding | Correction |
|------|------------------|------------|
| **Bounding Boxes** | bbox = box over the image | bbox = position of each token on page |
| **Labeling** | labels = actual words like "Walmart" | labels = BIO tags like `B-COMPANY` |
| **bbox scale** | pixel values can vary by brightness | scale to 0–1000 to handle any image size |
| **Loss vs. Metrics** | Used cross-entropy as metric | Cross-entropy is a loss; F1/precision/recall are metrics |
| **Input Shape** | Shape is 512x4 | Shape is `(batch_size, seq_length)` = (4, 512) |
| **Padding Mask** | Had 0s in valid token positions | All valid tokens = `1` in attention mask |

---

## 🧪 Drill-Down Exercise: Tagging + Token Inputs

### ✅ Task:

Given this line of OCR output:

```
QuickMart  2023-07-15  Total:  $35.99
```

Target:
- COMPANY: `QuickMart`
- DATE: `2023-07-15`
- TOTAL: `$35.99`

---

### 🔷 Step 1: Assign BIO Tags

| Token         | BIO Tag      |
|---------------|--------------|
| QuickMart     | ?
| 2023-07-15    | ?
| Total         | ?
| :             | ?
| $35.99        | ?

---

### 🔷 Step 2: Convert to `input_ids`

Assume:
```python
token_to_id = {
  "QuickMart": 1010,
  "2023-07-15": 1020,
  "Total": 1030,
  ":": 1040,
  "$35.99": 1050,
}

label_map = {
  "O": 0,
  "B-COMPANY": 1,
  "I-COMPANY": 2,
  "B-DATE": 3,
  "I-DATE": 4,
  "B-TOTAL": 5,
  "I-TOTAL": 6,
}
```

---

### 🔷 Step 3: Create the Input Dictionary

Fill in:
```python
input_ids = [...]
bbox = [ [..], [..], ... ]
attention_mask = [...]
labels = [...]
```

✔️ Try it yourself first! Then check your answer against the solution.

---

## 🔚 Next Steps

Once you're confident with inputs:
1. Build a custom `Dataset` class that yields this structure.
2. Add tokenizer alignment logic (handle token splits).
3. Set up training using `LayoutLMForTokenClassification`.
