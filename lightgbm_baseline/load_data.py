import json
from pathlib import PurePosixPath
from zipfile import ZipFile

import pandas as pd

from .config import STOCK_COLUMNS


def load_data(path):
    """Read only explicit official paths; never infer IDs as numbers."""
    with ZipFile(path) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f'ZIP CRC check failed: {damaged}')
        names = archive.namelist()
        def member(suffix):
            matches = [n for n in names if n == suffix or n.endswith('/' + suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected one ZIP member {suffix}, found {matches}")
            return matches[0]

        info = json.loads(archive.read(member('data/info.json')))
        stocks = info['Top50@Y']
        if not isinstance(stocks, list) or len(stocks) != 50 or not all(isinstance(s, str) for s in stocks):
            raise ValueError('Top50@Y must contain exactly 50 string stock IDs; inspect info.json schema')
        if len(set(stocks)) != 50:
            raise ValueError('Duplicate Top50 IDs')
        frames = []
        for stock in stocks:
            if PurePosixPath(stock).name != stock or '\\' in stock:
                raise ValueError('Invalid stock ID')
            with archive.open(member(f'data/market/{stock}.csv')) as stream:
                frame = pd.read_csv(stream, dtype={'stock_id': str})
            if 'stock_id' in frame and not frame.stock_id.eq(stock).all():
                raise ValueError(f'Stock ID mismatch in {stock}.csv')
            frame['stock_id'] = stock
            frames.append(frame[['date', 'stock_id', 'y', *STOCK_COLUMNS]])
        with archive.open(member('data/global/global_data.csv')) as stream:
            global_data = pd.read_csv(stream, usecols=['date', '^VIX'])
    return pd.concat(frames, ignore_index=True), global_data, stocks
