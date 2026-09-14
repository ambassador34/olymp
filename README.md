# Avito item dimensions submission

`src/train_and_predict.py` trains an ensemble for the Avito dimensions challenge and writes `output/submission.csv` in the required five-column format. It combines a CatBoost model over listing hierarchy, seller information, price, dates and parsed measurements with word/subword TF-IDF ridge regressors for the Russian title and description. Every target is fitted in `log1p` space to match Macro Log-MAE.

## Reproduce

1. Download `train.parquet` and `test.parquet` into `data/`.
2. Install dependencies: `python -m pip install pandas pyarrow scipy scikit-learn catboost`.
3. Run `python src/train_and_predict.py`.

The generated CSV is kept in `output/submission.csv` for direct upload.
