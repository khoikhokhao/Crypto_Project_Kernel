import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import pandas as pd

spec = importlib.util.spec_from_file_location('download', Path(__file__).resolve().parents[1] / 'scripts/download_ohlcv.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class CandleValidationTests(unittest.TestCase):
    def frame(self):
        return pipeline.normalize(pd.DataFrame([
            ['0','10','12','9','11','100','59999','1000','5','40','400','0'],
            ['120000','11','13','10','12','90','179999','1000','4','40','400','0'],
        ], columns=pipeline.COLS))

    def test_gap_detection_includes_boundaries(self):
        self.assertEqual(pipeline.missing_ranges(self.frame(), 0, 240000),
                         [[60000, 120000], [180000, 240000]])

    def test_header_and_numeric_conversion(self):
        frame = self.frame()
        header = pd.DataFrame([pipeline.COLS], columns=pipeline.COLS)
        raw = pd.DataFrame([
            ['0','10','12','9','11','100','59999','1000','5','40','400','0'],
        ], columns=pipeline.COLS)
        self.assertEqual(len(pipeline.normalize(pd.concat([header, raw]))), 1)
        pipeline.validate(frame)
        self.assertEqual(str(frame.open_time.dtype), 'int64')
        self.assertEqual(str(frame.close.dtype), 'float64')

    def test_reject_duplicate_and_bad_ohlc(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            pipeline.validate(pd.concat([frame, frame]))
        frame.loc[0, 'high'] = 8
        with self.assertRaisesRegex(ValueError, 'high'):
            pipeline.validate(frame)

    def test_reject_non_minute_and_incomplete_candle(self):
        frame = self.frame()
        frame.loc[0, 'open_time'] = 1
        with self.assertRaisesRegex(ValueError, 'aligned'):
            pipeline.validate(frame)
        frame = self.frame()
        frame.loc[0, 'close_time'] = 60000
        with self.assertRaisesRegex(ValueError, 'close time'):
            pipeline.validate(frame)

    def test_cached_subset_preserves_other_candles(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            path = out / 'BTCUSDT/1970-01.parquet'
            path.parent.mkdir()
            self.frame().to_parquet(path, index=False)
            with patch.object(pipeline, 'OUT', out), patch.object(pipeline, 'ROOT', out), patch.object(pipeline, 'api') as api:
                result = pipeline.month_job('BTCUSDT', pd.Timestamp('1970-01-01', tz='UTC'), 0, 60000)
            self.assertEqual(result['rows'], 1)
            self.assertEqual(pd.read_parquet(path).open_time.tolist(), [0, 120000])
            api.assert_not_called()

    def test_repair_missing_minute_from_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            path = out / 'BTCUSDT/1970-01.parquet'
            path.parent.mkdir()
            self.frame().to_parquet(path, index=False)
            recovered = self.frame().iloc[[0]].copy()
            recovered['open_time'] = 60000
            recovered['close_time'] = 119999
            with patch.object(pipeline, 'OUT', out), patch.object(pipeline, 'ROOT', out), patch.object(pipeline, 'api', return_value=recovered) as api:
                result = pipeline.month_job('BTCUSDT', pd.Timestamp('1970-01-01', tz='UTC'), 0, 180000)
            api.assert_called_once_with('BTCUSDT', 60000, 120000)
            self.assertEqual(result['missing_minutes'], 0)
            self.assertEqual(pd.read_parquet(path).open_time.tolist(), [0, 60000, 120000])


if __name__ == '__main__':
    unittest.main()
