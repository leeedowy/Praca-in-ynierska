import pandas as pd


def get_sentiment140(filename):
    df = pd.read_csv(filename, header=None, usecols=[0, 5], encoding='utf-8')

    df.columns = ['sentiment', 'text'] 
    df['sentiment'] = df['sentiment'] / 4

    return df['text'], df['sentiment']