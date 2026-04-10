# Text Encoding Reviews Analysis

This project collects Amazon product review data and demonstrates classic text encoding techniques for NLP feature engineering.

## Project Objective
- Scrape product reviews from Amazon search flows.
- Build a clean review dataset.
- Preprocess and clean review text.
- Compare text vectorization methods:
  - One-Hot Encoding (via `MultiLabelBinarizer`)
  - Bag of Words (`CountVectorizer`)
  - TF-IDF (`TfidfVectorizer`)
- Analyze matrix shapes, sparsity, and practical trade-offs.

## Project Structure
- `Notebook/Review_scrap.ipynb`: Scrapes review data and exports CSV.
- `Notebook/Text_featuring.ipynb`: Cleans text and applies encoding/vectorization methods.
- `Notebook/Comparison_Outcomes_Conclusion.md`: Summary of observations and conclusions.
- `data/Scrapedata.csv`: Sample scraped dataset (100 reviews).
- `main.py`: Minimal Python entry point placeholder.
- `pyproject.toml`: Basic Python project metadata.

## Dataset
The dataset contains columns such as:
- `Star Rating`
- `Review Title`
- `Review Description`

Current included sample: `100` reviews in `data/Scrapedata.csv`.

## Tech Stack
- Python
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- nltk
- requests, beautifulsoup4

## Workflow
1. Run `Notebook/Review_scrap.ipynb` to scrape and save reviews.
2. Use `Notebook/Text_featuring.ipynb` to:
   - clean and normalize text,
   - tokenize reviews,
   - generate OHE, BoW, and TF-IDF features,
   - inspect feature vocabulary and sparsity.
3. Refer to `Notebook/Comparison_Outcomes_Conclusion.md` for interpretation.

## Key Findings (Current Notebook Output)
- OHE matrix shape: `(100, 62)`
- BoW matrix shape: `(100, 167)`
- TF-IDF matrix shape: `(100, 167)`
- BoW and TF-IDF are highly sparse (about `97.89%` zeros).
- TF-IDF provides more informative weighting than plain BoW for many tasks.

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install jupyter pandas numpy scikit-learn nltk requests beautifulsoup4
```

3. Open notebooks:

```bash
jupyter notebook
```

## Notes
- This repository is learning/project-work focused and centered on notebook-based experiments.
- Some notebook paths are currently local-system specific and can be made relative for better portability.