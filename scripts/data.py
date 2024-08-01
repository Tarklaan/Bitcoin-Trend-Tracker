import mwclient
import time
from transformers import pipeline
from tqdm import tqdm
from statistics import mean
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

def find_sentiment(text):
    sent = sentiment_pipeline([text[:250]])[0]
    score = sent["score"]
    if sent["label"] == "NEGATIVE":
        score *= -1
    return score


site = mwclient.Site('en.wikipedia.org')
page = site.pages['Bitcoin']
revs = page.revisions()
revs = sorted(revs, key=lambda rev: rev["timestamp"])
sentiment_pipeline = pipeline("sentiment-analysis")
edits = {}
for rev in tqdm(revs, desc="Processing revisions"):
    date = time.strftime("%Y-%m-%d", rev["timestamp"])
    if date not in edits:
        edits[date] = dict(sentiments=list(), edit_count=0)

    edits[date]["edit_count"] += 1

    comment = rev.get("comment", "")
    edits[date]["sentiments"].append(find_sentiment(comment))

for key in edits:
    if len(edits[key]["sentiments"]) > 0:
        edits[key]["sentiment"] = mean(edits[key]["sentiments"])
        edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
    else:
        edits[key]["sentiment"] = 0
        edits[key]["neg_sentiment"] = 0
    del edits[key]["sentiments"]

edits_df = pd.DataFrame.from_dict(edits, orient="index")
edits_df.index = pd.to_datetime(edits_df.index)
dates = pd.date_range(start="2009-03-08",end=datetime.today())
edits_df = edits_df.reindex(dates, fill_value=0)
rolling_edits = edits_df.rolling(30, min_periods=30).mean()
rolling_edits = rolling_edits.dropna()

btc_ticker = yf.Ticker("BTC-USD")
btc = btc_ticker.history(period="max")
btc=btc[['Close']]

btc[f'SMA_7'] = btc['Close'].rolling(window=5).mean()
btc[f'EMA_7'] = btc['Close'].ewm(span=5, adjust=False).mean()
btc=btc[5:-1]

btc.index = btc.index.tz_localize(None)
data = pd.merge(btc, rolling_edits, how='inner', left_index=True, right_index=True)

data.to_csv("../data/data.csv")