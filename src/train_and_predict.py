"""Train a reproducible multimodal-ready tabular/text ensemble and write a submission."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'output'
TARGETS = ['weight', 'height', 'length', 'width']

def clean_text(frame):
    return (frame.title.fillna('') + ' ' + frame.description.fillna('')).str.lower().str.replace(r'\s+', ' ', regex=True)

def make_features(frame):
    x = frame.copy()
    text = clean_text(x)
    x['price_log'] = np.log1p(x.item_price.clip(lower=0))
    x['title_len'] = x.title.fillna('').str.len()
    x['description_len'] = x.description.fillna('').str.len()
    x['digit_count'] = text.str.count(r'\d')
    x['date_year'] = pd.to_datetime(x.order_date).dt.year
    x['date_month'] = pd.to_datetime(x.order_date).dt.month
    # Measurements explicitly stated in listings are unusually informative.
    for unit, pattern in {'cm': r'(\d+(?:[.,]\d+)?)\s*(?:см|cm)',
                          'mm': r'(\d+(?:[.,]\d+)?)\s*(?:мм|mm)',
                          'kg': r'(\d+(?:[.,]\d+)?)\s*(?:кг|kg)',
                          'g': r'(\d+(?:[.,]\d+)?)\s*(?:гр\b|г\b)'}.items():
        vals = text.str.findall(pattern).map(lambda z: [float(v.replace(',', '.')) for v in z])
        x[f'{unit}_n'] = vals.str.len()
        x[f'{unit}_max'] = vals.map(lambda z: max(z) if z else -1.0)
        x[f'{unit}_min'] = vals.map(lambda z: min(z) if z else -1.0)
    x['text_for_model'] = text
    return x

def score(y, p): return np.mean(np.abs(y - p))

def main():
    train = pd.read_parquet(DATA / 'train.parquet')
    test = pd.read_parquet(DATA / 'test.parquet')
    all_x = make_features(pd.concat([train.drop(columns=['real_weight','real_height','real_length','real_width']), test], ignore_index=True))
    x, xt = all_x.iloc[:len(train)].copy(), all_x.iloc[len(train):].copy()
    text, text_test = x.pop('text_for_model'), xt.pop('text_for_model')
    drop = ['item_id', 'image_name', 'title', 'description', 'order_date']
    x = x.drop(columns=drop); xt = xt.drop(columns=drop)
    cats = x.select_dtypes(include=['object','string']).columns.tolist()
    for c in cats:
        x[c] = x[c].fillna('__NA__').astype(str)
        xt[c] = xt[c].fillna('__NA__').astype(str)
    # Subword features preserve product models, dimensions and Russian inflections.
    vec_word = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=250000, sublinear_tf=True)
    vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=3, max_features=300000, sublinear_tf=True)
    text_train_matrix = hstack([vec_word.fit_transform(text), vec_char.fit_transform(text)]).tocsr()
    text_test_matrix = hstack([vec_word.transform(text_test), vec_char.transform(text_test)]).tocsr()
    predictions = {}
    for name in TARGETS:
        y = np.log1p(train['real_' + name].clip(lower=0).values)
        # Text ridge is a complementary high-cardinality product-name model.
        ridge = Ridge(alpha=7.0)
        ridge.fit(text_train_matrix, y)
        ridge_pred = ridge.predict(text_test_matrix)
        cb = CatBoostRegressor(
            loss_function='MAE', iterations=300, depth=9, learning_rate=.075,
            l2_leaf_reg=6, random_seed=2026, verbose=100, thread_count=-1,
            random_strength=.35, allow_writing_files=False)
        cb.fit(x, y, cat_features=cats)
        cb_pred = cb.predict(xt)
        # CatBoost handles hierarchy/price/condition; text improves exact product matching.
        predictions['target_' + name] = np.expm1(.72 * cb_pred + .28 * ridge_pred).clip(0.001, None)
    OUT.mkdir(exist_ok=True)
    result = pd.DataFrame({'item_id': test.item_id, **predictions})
    result.to_csv(OUT / 'submission.csv', index=False)
    print(result.describe().T)
    print(f'Wrote {OUT / "submission.csv"}')

if __name__ == '__main__':
    main()
