"""Binance USD-M perpetual 1m candles: verified archives + REST tail, UTC."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
from pathlib import Path
import threading
import time
import zipfile

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/binance_um'
OUT = ROOT / 'data/processed/binance_um/1m'
REPORTS = ROOT / 'data/reports'
COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_volume', 'trades', 'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']
FLOATS = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
          'taker_buy_base_volume', 'taker_buy_quote_volume']
LOCAL = threading.local()
LOG = logging.getLogger('download')


def get(url, params=None, missing_ok=False):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    for attempt in range(6):
        try:
            r = LOCAL.session.get(url, params=params, timeout=(15, 120))
            if r.status_code == 404 and missing_ok:
                return None
            if r.status_code in (418, 429):
                time.sleep(min(float(r.headers.get('Retry-After', 30)), 300))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f'Retry limit: {url}')


def normalize(frame):
    frame = frame.copy()
    # Some archive CSVs have a header, others do not.
    frame = frame[pd.to_numeric(frame.open_time, errors='coerce').notna()].copy()
    for col in ['open_time', 'close_time', 'trades']:
        frame[col] = pd.to_numeric(frame[col], errors='raise').astype('int64')
    for col in FLOATS:
        frame[col] = pd.to_numeric(frame[col], errors='raise').astype('float64')
    return frame.drop(columns='ignore').sort_values('open_time').reset_index(drop=True)


def validate(frame):
    if frame.empty:
        return
    if frame.isna().any().any():
        raise ValueError('Null values')
    if frame.open_time.duplicated().any():
        raise ValueError('Duplicate open timestamps')
    if (frame.open_time % 60000 != 0).any():
        raise ValueError('Timestamp not aligned to minute')
    if (frame.close_time != frame.open_time + 59999).any():
        raise ValueError('Invalid candle close time')
    if (frame[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError('Non-positive price')
    if (frame.high < frame[['open', 'close', 'low']].max(axis=1)).any():
        raise ValueError('Invalid high price')
    if (frame.low > frame[['open', 'close', 'high']].min(axis=1)).any():
        raise ValueError('Invalid low price')
    if (frame[['volume', 'quote_volume', 'trades', 'taker_buy_base_volume',
               'taker_buy_quote_volume']] < 0).any().any():
        raise ValueError('Negative volume/trades')


def archive(symbol, period, kind):
    filename = f'{symbol}-1m-{period}.zip'
    target = RAW / symbol / kind / filename
    url = f'https://data.binance.vision/data/futures/um/{kind}/klines/{symbol}/1m/{filename}'
    checksum_file = target.with_suffix('.zip.CHECKSUM')
    # Cached immutable archive bytes are rechecked before every use.
    if not target.exists() or not checksum_file.exists():
        check = get(url + '.CHECKSUM', missing_ok=True)
        if check is None:
            return None
        response = get(url)
        expected = check.text.split()[0]
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {url}')
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix('.zip.tmp')
        tmp.write_bytes(response.content)
        tmp.replace(target)
        checksum_file.write_text(check.text, encoding='utf-8')
    if hashlib.sha256(target.read_bytes()).hexdigest() != checksum_file.read_text().split()[0]:
        raise ValueError(f'Cached checksum mismatch: {target}')
    with zipfile.ZipFile(target) as z:
        names = [n for n in z.namelist() if n.endswith('.csv')]
        if len(names) != 1:
            raise ValueError(f'Unexpected archive contents: {target}')
        with z.open(names[0]) as f:
            frame = pd.read_csv(f, header=None, names=COLS, dtype=str)
    return normalize(frame)


def api(symbol, start, end):
    parts = []
    cursor = start
    while cursor < end:
        rows = get('https://fapi.binance.com/fapi/v1/klines', params={
            'symbol': symbol, 'interval': '1m', 'startTime': cursor,
            'endTime': end - 1, 'limit': 1000}).json()
        if not rows:
            break
        frame = normalize(pd.DataFrame(rows, columns=COLS))
        parts.append(frame)
        next_cursor = int(frame.open_time.iloc[-1]) + 60000
        if next_cursor <= cursor:
            raise ValueError('API cursor did not advance')
        cursor = next_cursor
        time.sleep(0.15)
    return pd.concat(parts, ignore_index=True) if parts else normalize(pd.DataFrame(columns=COLS))


def missing_ranges(frame, start, end):
    present = set(frame.open_time)
    missing = [t for t in range(start, end, 60000) if t not in present]
    ranges = []
    for t in missing:
        if ranges and ranges[-1][1] == t:
            ranges[-1][1] = t + 60000
        else:
            ranges.append([t, t + 60000])
    return ranges


def stamp(ms):
    return pd.Timestamp(ms, unit='ms', tz='UTC').isoformat()


def month_job(symbol, month, start, end):
    left = max(start, int(month.timestamp() * 1000))
    next_month = month + pd.offsets.MonthBegin(1)
    right = min(end, int(next_month.timestamp() * 1000))
    path = OUT / symbol / f'{month:%Y-%m}.parquet'
    source = 'cached parquet'
    retained = None
    if path.exists():
        frame = pd.read_parquet(path)
        frame = frame.drop(columns=['timestamp'], errors='ignore')
        retained = frame[(frame.open_time < left) | (frame.open_time >= right)].copy()
        frame = frame[(frame.open_time >= left) & (frame.open_time < right)].copy()
    else:
        frame = archive(symbol, f'{month:%Y-%m}', 'monthly')
        source = 'monthly archive'
        if frame is None:
            source = 'daily archives + REST'
            parts = []
            day = month
            while int(day.timestamp() * 1000) < right:
                day_end = int((day + pd.Timedelta(days=1)).timestamp() * 1000)
                if day_end <= end and day_end > left:
                    daily = archive(symbol, f'{day:%Y-%m-%d}', 'daily')
                    if daily is not None:
                        parts.append(daily)
                day += pd.Timedelta(days=1)
            frame = pd.concat(parts, ignore_index=True) if parts else normalize(pd.DataFrame(columns=COLS))
        frame = frame[(frame.open_time >= left) & (frame.open_time < right)].copy()
    validate(frame)
    gaps = missing_ranges(frame, left, right)
    if gaps:
        source += ' + REST repair'
        recovered = [api(symbol, a, b) for a, b in gaps]
        frame = pd.concat([frame, *recovered], ignore_index=True)
        frame = frame[(frame.open_time >= left) & (frame.open_time < right)]
        frame = frame.sort_values('open_time').reset_index(drop=True)
    validate(frame)
    gaps = missing_ranges(frame, left, right)
    stored = pd.concat([retained, frame], ignore_index=True) if retained is not None else frame.copy()
    stored = stored.sort_values('open_time').reset_index(drop=True)
    validate(stored)
    stored.insert(0, 'timestamp', pd.to_datetime(stored.open_time, unit='ms', utc=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.parquet.tmp')
    stored.to_parquet(tmp, index=False, compression='zstd')
    tmp.replace(path)
    result = {'symbol': symbol, 'month': f'{month:%Y-%m}', 'rows': len(frame),
              'expected_rows': (right-left)//60000, 'first_utc': stamp(int(frame.open_time.min())) if len(frame) else None,
              'last_utc': stamp(int(frame.open_time.max())) if len(frame) else None,
              'missing_minutes': sum((b-a)//60000 for a,b in gaps),
              'gaps': [{'start_utc': stamp(a), 'end_exclusive_utc': stamp(b)} for a,b in gaps],
              'source': source, 'path': str(path.relative_to(ROOT))}
    LOG.info('%s %s rows=%d missing=%d', symbol, result['month'], len(frame), result['missing_minutes'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2021-01-01', help='UTC inclusive')
    parser.add_argument('--end', help='UTC exclusive; default exchange server current minute')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT','ETHUSDT','XRPUSDT','BNBUSDT','DOGEUSDT'])
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    (ROOT / 'logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / 'logs/download.log', encoding='utf-8')])
    server_ms = int(get('https://fapi.binance.com/fapi/v1/time').json()['serverTime'])
    closed_end = server_ms // 60000 * 60000
    start_ts = pd.Timestamp(args.start, tz='UTC')
    start = int(start_ts.timestamp()*1000)
    end = min(int(pd.Timestamp(args.end, tz='UTC').timestamp()*1000), closed_end) if args.end else closed_end
    if start % 60000 or end % 60000 or start >= end:
        parser.error('Require minute-aligned start < end')
    months = pd.date_range(start_ts.normalize().replace(day=1), pd.Timestamp(end-1, unit='ms', tz='UTC'), freq='MS')
    results, errors = [], []
    LOG.info('Snapshot UTC [%s, %s), %d months, %s', stamp(start), stamp(end), len(months), args.symbols)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(month_job, s, m, start, end): (s, str(m)) for s in args.symbols for m in months}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as exc:
                s, m = futures[f]
                LOG.exception('FAILED %s %s', s, m)
                errors.append({'symbol': s, 'month': m, 'error': str(exc)})
    summaries = []
    for s in args.symbols:
        rows = sorted([r for r in results if r['symbol']==s], key=lambda r:r['month'])
        summaries.append({'symbol':s, 'rows':sum(r['rows'] for r in rows),
                          'expected_rows':(end-start)//60000, 'first_utc':rows[0]['first_utc'] if rows else None,
                          'last_utc':rows[-1]['last_utc'] if rows else None,
                          'missing_minutes':sum(r['missing_minutes'] for r in rows),
                          'failed_partitions':sum(e['symbol']==s for e in errors)})
    report = {'exchange':'Binance', 'market':'USD-M perpetual', 'interval':'1m',
              'start_inclusive_utc':stamp(start), 'end_exclusive_utc':stamp(end),
              'captured_server_time_utc':stamp(server_ms), 'summaries':summaries,
              'partitions':sorted(results, key=lambda r:(r['symbol'],r['month'])), 'errors':errors}
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / 'quality_latest.json'
    tmp = report_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, indent=2), encoding='utf-8')
    tmp.replace(report_path)
    pd.DataFrame(summaries).to_csv(REPORTS / 'summary.csv', index=False)
    print(json.dumps(summaries, indent=2), flush=True)
    if errors or any(s['rows'] != s['expected_rows'] for s in summaries):
        raise SystemExit('INCOMPLETE: inspect data/reports/quality_latest.json')
    LOG.info('COMPLETE: all requested minute timestamps present, validated, no duplicates.')


if __name__ == '__main__':
    main()
