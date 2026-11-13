# Grid search nhanh

`bt_crypto_scan.scan_risk_grid` quét một chuỗi entry signal với nhiều tổ hợp TP, SL, max-hold bằng Numba và đa luồng. Nó dùng cùng quy tắc với `build_targets` trong notebook BTC alpha38 và accounting của `bt_crypto`: quyết định tại Close, fill next observed Open, fixed margin, actual fill, phí notional từng leg, slippage và MTM Close. Không tạo trade/bar ledger cho mỗi cấu hình.

## Cài đặt

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-grid.txt
```

Nếu notebook đang mở, restart kernel `.venv` sau khi cài. Lần gọi đầu có chi phí biên dịch. `requirements.lock.txt` là snapshot môi trường dự án đã cài đầy đủ.

## API cho alpha khác

```python
import pandas as pd
from bt_crypto import BacktestConfig
from bt_crypto_scan import scan_risk_grid

# df: một asset, OHLCV UTC 5m, index là thời gian mở nến.
# entry_signals: Series -1/0/+1, cùng index, quyết định sau Close.
risk_grid = pd.DataFrame({"tp": [0.02, 0.04], "sl": [0.01, 0.02], "max_hold": [200, 500]})
config = BacktestConfig(asset="BTCUSDT", timeframe="5min", timestamp_label="open",
                        contract_type="linear_usdt", fx_vnd_per_usdt=25000,
                        slippage_bps=1.0)
results = scan_risk_grid(ohlcv=df, entry_signals=entry_signals, risk_grid=risk_grid,
                         config=config, workers=8, batch_size=256)
```

`risk_grid` là các dòng tổ hợp đã tạo, không tự tạo Cartesian product. Signals chỉ được xét khi flat; khi đang giữ, entry signals bị bỏ qua. TP/SL xét high/low của nến kể cả nến entry, tại Close tạo target exit; exit khớp Open quan sát tiếp theo. `max_hold` đếm số nến quan sát, gồm entry bar. Cùng nến chạm cả TP/SL vẫn chỉ có một exit next Open. Không tự đóng vị thế ở cuối dữ liệu. Với logic target, exit hoặc execution khác, không dùng API này như thể tương đương; cần scanner riêng và parity test.

`results` có net PnL, annualized simple return, Sharpe từ UTC daily PnL, MDD tính trên vốn ban đầu, số closed/open trade, win rate, average PnL/trade, phí, slippage, margin turnover và missing bars. Sharpe loại ngày không đủ dữ liệu và ngày có carry qua gap theo `bt_crypto`. Undefined trả NaN và có `sharpe_reason`. Mọi kết quả trong RAM.

## Notebook BTC alpha38

Cell grid cuối của `research/btc_alpha38_vwpd_5m.ipynb` tự ghép ATR ngắn < dài, tính VWPD một lần cho mỗi cặp ATR, thử thresholds rồi quét risk grid. Chỉnh `GRID`, `GRID_START_UTC`, `GRID_END_UTC`, `WORKERS` và `BATCH_SIZE` trong cell. Chạy các cell phía trên trước để có data, functions và config. Nếu VS Code đang giữ bản notebook cũ trong bộ nhớ, tải lại file trước khi chạy.

Các helper `load_bars`, `features_by_segment`, `build_targets`, `run_grid` nằm trong `research/alpha_helpers.py`. Công thức feature và điều kiện entry vẫn định nghĩa trực tiếp trong notebook; cả backtest và grid dùng chung hai hàm đó. `build_targets` áp dụng exit TP/SL/max-hold theo quy ước của scanner, không phải engine tổng quát cho mọi loại exit.

`grid_results` giữ mọi cấu hình theo thứ tự quét. Sau khi chạy xong, cell lưu một CSV vào `research/grid_results/` gồm tham số, Sharpe, MDD trên vốn ban đầu, annualized simple return và số trade đóng; timestamp UTC giúp giữ các lần chạy trước. Không lọc hay xếp hạng. Sau khi chọn cấu hình, nhập tham số vào cell baseline và chạy lại bằng `bt_crypto` để nhận trade ledger, equity và diagnostics chi tiết. Kết quả tìm kiếm là in-sample, cần kiểm tra ngoài mẫu riêng.

## Giới hạn và kiểm tra

- Fast scanner **chưa hỗ trợ funding** và từ chối funding settings; kết quả ghi `funding_included=False`. Không diễn giải PnL là sau mọi chi phí.
- Hỗ trợ một asset linear USDT; không portfolio, liquidation, order book hay execution mô phỏng thực tế.
- `workers` tối đa bằng số thread Numba hiện có. Không tự mở thêm process. `batch_size` giới hạn số cấu hình trong mỗi lần gọi kernel.
- `tests/test_bt_crypto_scan.py` đối chiếu scanner và notebook grid với `bt_crypto` trên synthetic OHLCV, gồm gaps, missing days, terminal open, phí/slippage và nhiều TP/SL/max-hold. Thực tế cũng đã đối chiếu trên BTC 5m năm 2025.
- Các tính toán dùng float64, không bật fastmath. Thứ tự cộng có thể tạo sai số số thực nhỏ; kiểm tra dùng tolerance.
