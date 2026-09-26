📌 SpectraCompress — Day 1 Short Notes

Main Goal: Raw MNIST data → validate → train/validation split → simple Top-K compression baseline → reconstruction error.

1. Data Contract
Data ची shape, dtype, valid range define केला.
Unsupervised project असल्यामुळे target_column = None.
2. Schema Validation

SchemaValidator ने check केलं:

Shape
Dtype
NaN / Inf
Valid value range
Duplicate rows

Invalid data → InputValidation.

3. Train / Validation Split

RowSplitter:

Raw rows shuffle करून 80% train / 20% validation
seed=42 → same split पुन्हा मिळतो.
Train आणि validation मध्ये row overlap नाही.
No preprocessing before split → leakage avoid.
4. Top-K Variance Baseline

Train data वर:

variance प्रत्येक pixel ची
        ↓
highest variance pixels
        ↓
top_k_indices

उदा. TOP_K=100 → 784 pixels मधून 100 select.

Important: variance फक्त train data वर calculate.

5. Transform

top_k_indices वापरून नवीन data मधून तेच columns select:

(N, 784) → (N, K)

Validation साठी पुन्हा feature selection नाही.

6. Reconstruction Error

Original आणि reconstructed data मधला difference measure केला.

Per-row Frobenius:

$$ \sqrt{\sum_j (x_j-\hat{x}_j)^2} $$

axis=1 → प्रत्येक image/row चा error.

7. Day 1 Tests

6 Pytest tests:

Wrong schema reject
Same seed → same split
Train/val overlap नाही
Baseline train-only behavior
Reconstruction error ≥ 0 आणि expected
Top-K selection synthetic data वर correct
🔑 Day 1 चा One-Line Memory Hook

“Validate → Split → Train वर Variance → Top-K Pixels → Transform → Reconstruction Error → Tests.”