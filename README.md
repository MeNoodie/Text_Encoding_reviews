<h1 align="center">Sentiment & Issue Detection</h1>

<p align="center">
  Bulk review analysis with ML + notebook workflows
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13-blue"/>
  <img src="https://img.shields.io/badge/notebook-workflow-orange"/>
  <img src="https://img.shields.io/badge/model-logistic_regression-brightgreen"/>
</p>

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Results](#model-results)
- [Usage](#usage)
- [Example Prediction](#example-prediction)
- [Tech Stack](#tech-stack)
- [Notes](#notes)

---

## Features

- Scrapes and stores Amazon review data for sentiment analysis experiments.
- Cleans review text with normalization, stopword removal, and lemmatization.
- Creates sentiment labels from star ratings: positive, neutral, and negative.
- Builds text features with One-Hot Encoding, Bag of Words, and TF-IDF.
- Trains a sentiment classification model from scratch using extracted sentiment-word features.
- Compares Logistic Regression, Multinomial Naive Bayes, and Bernoulli Naive Bayes.
- Demonstrates prediction on a new user review comment.

## Project Structure

- `Notebook/Review_scrap_before_2000_reviews.ipynb`: Scrapes reviews and exports the larger dataset.
- `Notebook/model.ipynb`: Main notebook for preprocessing, encoding, model training, model comparison, and example prediction.
- `Notebook/Text_featuring.ipynb`: Earlier text encoding notebook based on the smaller sample dataset.
- `Notebook/Comparison_Outcomes_Conclusion.md`: Legacy notes from the earlier 100-review experiment.
- `data/Reviews.csv`: Current dataset with 1500 reviews.
- `data/Scrapedata.csv`: Older sample dataset with 100 reviews.

## Dataset

The datasets contain these columns:

- `Star Rating`
- `Review Title`
- `Review Description`

Current working dataset:

- `data/Reviews.csv`: `1500` rows

Processed modeling output in `Notebook/model.ipynb`:

- Final processed dataset shape: `(898, 4)`

## Model Results

The sentiment classification section in `Notebook/model.ipynb` uses:

- `sentiment_words` as the feature
- `sentiment_label` as the target
- `TfidfVectorizer(ngram_range=(1, 2))` for feature extraction

Saved notebook accuracies:

- Logistic Regression: `0.8944`
- Multinomial Naive Bayes: `0.8873`
- Bernoulli Naive Bayes: `0.8662`

Best performing model in the current notebook:

- Logistic Regression

Text encoding output shapes:

- One-Hot Encoding: `(898, 30)`
- Bag of Words: `(898, 30)`
- TF-IDF: `(898, 30)`

## Usage

1. Create and activate a virtual environment.
2. Install the required packages.
3. Launch Jupyter Notebook.
4. Run the scraping notebook if you want to collect data.
5. Run `Notebook/model.ipynb` for preprocessing, feature engineering, model comparison, and prediction.

```bash
pip install jupyter pandas scikit-learn nltk requests beautifulsoup4
jupyter notebook
```

Suggested notebook order:

```text
Notebook/Review_scrap_before_2000_reviews.ipynb
Notebook/model.ipynb
```

## Example Prediction

The notebook includes a demo prediction on a new user comment.

- Extracted sentiment words: `best`, `good`, `great`, `issue`, `perfect`, `stable`, `strong`
- Predicted sentiment: `positive`

This gives a simple end-to-end example of how the trained pipeline can classify new review text.

## Tech Stack

- Python
- Jupyter Notebook
- pandas
- scikit-learn
- nltk
- requests
- beautifulsoup4

## Notes

- `Notebook/model.ipynb` is the main notebook representing the latest project work.
- The older comparison markdown and `Text_featuring.ipynb` reflect the previous smaller dataset workflow.
- NLTK resources may need to be downloaded when running the notebook for the first time.
