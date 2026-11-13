"""Read only requested columns/time range from monthly Parquet partitions."""
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds

DATA = Path(__file__).resolve().parents[1] / 'data/processed/binance_um/1m'


def load_ohlcv(symbol, start=None, end=None, columns=None):
    """UTC start inclusive, end exclusive. Example symbol: BTCUSDT."""
    symbol = symbol.upper().replace('/', '')
    if symbol not in {'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT'}:
        raise ValueError('Unsupported symbol')
    dataset = ds.dataset(DATA / symbol, format='parquet', exclude_invalid_files=True)
    predicate = None
    for date, op in [(start, 'ge'), (end, 'lt')]:
        if date is None:
            continue
        value = pd.Timestamp(date)
        value = value.tz_localize('UTC') if value.tzinfo is None else value.tz_convert('UTC')
        ms = value.value // 1_000_000
        expr = ds.field('open_time') >= ms if op == 'ge' else ds.field('open_time') < ms
        predicate = expr if predicate is None else predicate & expr
    selected = columns or ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    frame = dataset.to_table(columns=selected, filter=predicate).to_pandas()
    sort_column = 'timestamp' if 'timestamp' in selected else 'open_time'
    if sort_column in selected:
        frame = frame.sort_values(sort_column).reset_index(drop=True)
    return frame
