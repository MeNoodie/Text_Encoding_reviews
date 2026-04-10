# Comparison Outcomes and Conclusion (Based on `Text_featuring.ipynb`)

This report is prepared from the outputs in `Text_Encoding/Notebook/Text_featuring.ipynb` 

##  Comparison Analysis (OHE vs BoW vs TF-IDF)

### A. Output Comparison Table

| Feature Method | Matrix Shape | What Value Means | Weighting Style | Key Observation from Notebook |
|---|---:|---|---|---|
| OHE (`MultiLabelBinarizer`) | `(100, 62)` | `1` = present, `0` = absent | No frequency/importance weighting | Vocabulary sample contains characters like `' '`, `'0'`, `'1'` etc., so current OHE is character-level in this notebook output. |
| BoW (`CountVectorizer`) | `(100, 167)` | Integer token count | Frequency only | Captures repeated words but treats all words equally except by count. |
| TF-IDF (`TfidfVectorizer`) | `(100, 167)` | Weighted score | Frequency + rarity across documents | Same vocabulary size as BoW, but rare informative words get higher scores than very common words. |

### B. Which words matter more in TF-IDF (and why)

From notebook word-frequency output, very common words include `phone`, `good`, `product`, `Great`, `Very`, etc.  
In TF-IDF, very common words usually receive lower weight because they appear in many titles and are less discriminative.  
Less common/context-specific terms get relatively higher TF-IDF values because they help distinguish one title from another.

## Sparse Matrix Analysis

### A. Shapes

- OHE shape: `(100, 62)`
- BoW shape: `(100, 167)`
- TF-IDF shape: `(100, 167)`

### B. Sparsity from notebook output

Notebook prints:
- OHE Matrix Sparsity: `0.21725806451612903`
- BoW Matrix Sparsity: `0.021137724550898202`
- TF-IDF Matrix Sparsity: `0.021137724550898202`

In the code, this value is actually **non-zero ratio (density)**, not zero ratio.  
So the **percentage of zeros** is:

- OHE zeros = `1 - 0.2172580645 = 0.7827419355` -> **78.27%**
- BoW zeros = `1 - 0.0211377246 = 0.9788622754` -> **97.89%**
- TF-IDF zeros = `1 - 0.0211377246 = 0.9788622754` -> **97.89%**

### C. Why sparse matrices are efficient

- Most entries are zero, especially in BoW/TF-IDF, so sparse storage avoids saving full dense arrays.
- Memory usage is reduced by storing only non-zero positions and values.
- Many ML libraries optimize sparse operations, improving training/inference speed.

## Real-world Questions

### 1) Why Bag of Words fails for semantic meaning

- BoW ignores context and word order.
- It cannot capture synonyms well (for example: `good` vs `excellent` may be treated separately).
- It may fail on phrases where meaning depends on sequence (e.g., negation like "not good").

### 2) When to use BoW and when to use TF-IDF

- Use **BoW** when:
  - You need a simple baseline.
  - Dataset/domain is small and vocabulary is controlled.
  - Raw frequency itself is meaningful.
- Use **TF-IDF** when:
  - You want better discriminative text features.
  - Common words should be down-weighted.
  - You are building standard text classification/retrieval pipelines.

### 3) Limitations of TF-IDF in real applications

- Still ignores deep semantics and word order.
- Cannot represent contextual meaning of words (polysemy).
- Rare noisy terms can get high weight.
- Performance can degrade when language is highly contextual or domain shifts occur.

## Conclusion

- For this notebook, **BoW and TF-IDF** produce richer word-level feature spaces (`167` features) compared to current OHE output (`62` features).
- **TF-IDF is generally more useful** than BoW for modeling because it reduces the dominance of very common words and emphasizes informative terms.
- Feature matrices are highly sparse (especially BoW/TF-IDF, ~97.89% zeros), so sparse matrix representation is appropriate and efficient.
- Current OHE implementation appears character-level in output; converting OHE to proper token-level encoding would make comparison cleaner and more consistent.
