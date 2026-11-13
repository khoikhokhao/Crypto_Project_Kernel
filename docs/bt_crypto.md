# bt_crypto: một asset, linear USDT futures

Module ở thư mục gốc: `bt_crypto.py`. Chỉ hỗ trợ `contract_type="linear_usdt"`; không dùng cho inverse/coin-margined. Core chỉ cần NumPy và pandas; `result.plot()` cần Matplotlib. Không ghi kết quả ra file, không gọi mạng.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-backtest.txt
# Nếu mở/chạy notebook ví dụ (chọn kernel .venv trong VS Code):
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_bt_crypto.py -v
```

## API và schema

```python
from bt_crypto import BacktestConfig, backtest

config = BacktestConfig(
    asset="BTCUSDT", timeframe="15min", timestamp_label="open",
    contract_type="linear_usdt",
    fx_vnd_per_usdt=25_000,  # Giả định cố định do researcher chọn, không phải FX thị trường
    slippage_bps=1.0,
    # Mặc định: fee_rate_per_side=0.0005 (0,05% mỗi chiều), initial capital và margin budget 10 tỷ VND, leverage=1
)
result = backtest(ohlcv=df, target_position=position_series, config=config)
display(result.summary())  # Metrics và lý do undefined
display(result.trades)
result.plot()
```

`df` có **DatetimeIndex UTC**, tăng dần, duy nhất, không NaT; nhãn là **đầu nến**. Cột bắt buộc lowercase: `open, high, low, close, volume`, numeric, hữu hạn; giá dương, OHLC hợp lệ và volume không âm. Một lần chạy chỉ một asset. Nếu có `asset`/`symbol`, phải duy nhất và khớp config. Không truyền bảng nhiều asset.

Timeframe là khoảng cố định chia hết một ngày UTC, ví dụ `1min`, `15min`, `4h`, `1D`; nhãn phải căn lưới UTC epoch. Không hỗ trợ tuần/tháng hoặc nến vượt ngày. Nếu có `open_time`/`close_time`, chúng được kiểm tra theo Unix milliseconds của kho Binance (close inclusive = open + timeframe - 1ms). Với OHLCV đã resample, bỏ metadata timestamp của nến 1m hoặc tính lại đúng timeframe. Nếu không có metadata, `timestamp_label="open"` là cam kết của caller: không thể suy ra nhãn open/close chỉ từ giá OHLC.

`position_series` là Series numeric chỉ gồm `-1, 0, +1`, có **chính xác cùng index** với OHLCV. NaN, bool, index lệch hoặc giá trị khác gây lỗi. Không normalize, reindex, shift đầu vào hay fill âm thầm. Notebook tự tính features và chủ động chọn xử lý warmup.

Xem notebook [btc_alpha38_vwpd_5m.ipynb](../research/btc_alpha38_vwpd_5m.ipynb), sử dụng `research.load_data.load_ohlcv` hiện có. Khi resample, notebook chỉ giữ bucket có đủ nến 1m và không tạo nến giả.

## Timing, sizing và chi phí

- Bắt đầu flat. Target quyết định sau Close[t] → khớp tại **Open quan sát kế tiếp**, dùng chính timeframe đầu vào.
- Đổi hướng: exit trước, entry ngược sau tại cùng Open, tính hai legs. Cùng hướng: giữ entry anchor và quantity; không rebalance/pyramiding.
- Target ở bar cuối không được khớp. Vị thế còn mở được MTM tại Close cuối; không giả exit, không phí exit.
- Nếu thiếu bar: giữ vị thế và pending target đến Open quan sát tiếp theo; ghi `result.gaps` và `ledger.missing_bars_before`. Không forward-fill/synthetic bars.

VND là currency báo cáo, FX VND/USDT **bắt buộc khai báo và cố định** suốt experiment:

```text
entry_notional_usdt = fixed_margin_budget_vnd / fx_vnd_per_usdt * leverage
BUY fill  = observed_open * (1 + slippage_bps / 10000)
SELL fill = observed_open * (1 - slippage_bps / 10000)
quantity  = entry_notional_usdt / actual_entry_fill
fee_usdt  = abs(quantity) * actual_fill * fee_rate_per_side
gross_pnl_usdt = direction * quantity * (actual_exit_fill - actual_entry_fill)
```

Quantity dương, direction tách riêng. Cho phép fractional quantity, giả định đủ thanh khoản. Mỗi entry/flip dùng lại ngân sách gốc, không dùng current equity. Fee mặc định 0,05% mỗi chiều, tương đương 0,1% roundtrip khi entry và exit có cùng notional. Fee tính **actual traded notional** từng leg, quantity đã chứa leverage: **không nhân leverage lần nữa**. Có thể cấu hình `fee_rate_per_side` riêng cho experiment; fee/slippage bằng 0 chỉ khi caller cấu hình rõ.

Margin 10 tỷ VND, leverage 5, fee 0,05% mỗi chiều → entry notional 50 tỷ, entry fee **25 triệu**. Exit cùng giá: exit fee 25 triệu, tổng 50 triệu. Entry 100 → exit 110: exit fee **27,5 triệu**, tổng **52,5 triệu**. Các trường hợp này có unit tests tính tay.

Slippage nằm trong actual fill, vì vậy **gross PnL đã chịu slippage**. `slippage_*` là attribution dương, không phải một khoản cash charge trừ lần nữa. `reference_gross_profit_before_slippage_vnd = gross_profit_vnd + total_slippage_vnd` dùng cùng actual quantity để đối chiếu với Open không slippage.

## Funding riêng

OHLCV không chứa funding. Khi không truyền events, `total_funding_cashflow_vnd=None` kèm lý do và report/plot ghi **funding NOT included**. Net ở đây là sau phí và fill slippage, chưa bao gồm funding.

```python
funding = pd.DataFrame(
    {"rate": rates, "mark_price": settlement_mark_prices},
    index=settlement_times_utc,
)
config = dataclasses.replace(
    config,
    funding_convention="positive_rate_longs_pay",
    funding_at_open="before_fill",  # Hoặc after_fill: researcher xác minh theo nguồn
)
result = backtest(ohlcv=df, target_position=position_series,
                  config=config, funding_events=funding)
```

Index funding UTC, tăng dần, duy nhất; timestamps trong `[first_open, final_bar_end)`. Bắt buộc truyền rate có dấu và **mark price tại thanh toán**, không suy từ Open/Close tương lai. Không tự chia rate hằng năm/hằng ngày: caller chuyển thành rate đúng từng event của nguồn.

`cashflow_usdt = -direction_at_settlement * quantity * mark_price * rate`. Cashflow dương là nhận, âm là trả; rate dương làm LONG trả, SHORT nhận. `funding_at_open` bắt buộc khi có events, xác định vị thế đủ điều kiện nếu settlement trùng Open: `before_fill` dùng vị thế cũ, `after_fill` dùng vị thế sau orders. Đây là quy ước experiment do caller khai báo, không tự khẳng định quy tắc thực tế của một sàn.

Events trong gap vẫn thanh toán trên vị thế đang giữ, trước next observed fill. Thời gian thật nằm trong `result.events`; phần cash phát sinh trong gap được đưa vào bar quan sát kế tiếp. MTM từng ngày trong gap không suy diễn. Empty events chỉ là caller chủ động cung cấp bảng rỗng; engine **không xác minh coverage funding**, kể cả khi có events.

## Accounting và output

```text
realized_cash_pnl = cumulative realized gross - all entry/exit fees + funding cashflows
equity_vnd = initial_capital_vnd + (realized_cash_pnl_usdt + unrealized_gross_pnl_usdt) * FX
bar_pnl_vnd = equity change, first bar measured against initial capital
```

Bao gồm Open→Close ở entry bar, gap khi đang holding, không tính gap trước entry, không đếm lại PnL khi exit/flip. Timestamp index của ledger vẫn là **bar open**; `mark_time` là biên kết thúc bar (exclusive), tại đó ghi Close MTM.

| Output | Nội dung |
| --- | --- |
| `metrics` | Dict, tỷ lệ dạng fraction, tiền VND, holding duration là Timedelta |
| `metric_reasons` | Lý do cho metric None |
| `trades` | Một dòng mỗi vòng mở→đóng hoặc vị thế terminal open; net của open bao gồm MTM, entry fee và funding đã ghi nhận |
| `ledger` | Mỗi bar: target, executed target, vị thế sau Open, quantity, anchor, cash/unrealized, equity, PnL, fees, slippage, funding, turnover, gap và diagnostics |
| `events` / `fills` | Events entry/exit/funding theo thứ tự xử lý; `fills` chỉ hai loại order legs. Tiền trong events là USDT |
| `daily_equity` | UTC calendar days, equity cuối ngày quan sát và daily PnL; ngày mất toàn bộ dữ liệu giữ NaN |
| `gaps` / `diagnostics` | Khoảng thiếu, funding coverage, terminal target và cảnh báo vốn |

Đối chiếu: `sum(trades.net_pnl_vnd) = sum(ledger.bar_pnl_vnd) = final_equity - initial_capital`; `sum(events.cash_change_usdt) = final_realized_cash_pnl_usdt`. Mỗi bar: `bar_pnl_vnd = bar_gross_pnl_vnd - fees_vnd + funding_cashflow_vnd`.

## Metrics

- Gross/Net profit gồm closed PnL và terminal open MTM. Gross theo actual fills, net trừ fees và cộng funding được cung cấp. Total slippage chỉ attribution; total funding là **cashflow có dấu**.
- Annualized **simple** return = `net_profit / initial_capital / elapsed_years`, năm = 365 ngày, elapsed từ first Open đến final bar end, kể cả gaps. Không CAGR/compounding. 5 năm đúng 365 ngày/năm, lãi 50 tỷ trên vốn 10 tỷ → 100%/năm.
- MDD VND từ running peak **có gồm vốn ban đầu**; MDD fraction = `MDD_VND / initial_capital`, không chia peak equity. Đo trên các Close quan sát và baseline ban đầu, không đo intrabar excursions.
- Sharpe = mean/std(ddof=1) của daily net PnL / **vốn ban đầu**, nhân sqrt(365), risk-free = 0. Chỉ dùng UTC days đủ toàn bộ bars, loại partial boundary days, ngày thiếu bar và ngày chứa carry PnL qua gap. Ngày đủ dữ liệu flat PnL=0 vẫn được dùng. Không đủ hai ngày hoặc std=0 → None và lý do.
- Win rate = tỷ lệ **closed trades net>0**; net=0 không win. Average net PnL/trade và average holding duration chỉ tính closed trades; duration là thời gian thực entry→exit, bao gồm gaps.
- Average daily **margin turnover on initial** = mean theo **ngày có bars quan sát**, gồm partial boundary days, loại ngày mất toàn bộ dữ liệu, của `sum(abs(q)*fill*FX/leverage)/initial_capital`. Cộng cả entry/exit, vì vậy flip có hai legs. Đây là margin turnover, không phải notional turnover. Chính sách ngày có ghi trong diagnostics.

## Giới hạn

Không liquidation/margin call, kiểm tra sức mua, lot/step size, order book, queue, latency, TP/SL, max-hold, funding-data completeness, portfolio hay optimizer. Entry budget không bị giảm khi equity thấp. Flags `equity_non_positive` và `margin_exceeds_equity` chẩn đoán tại observed Open/Close và mark price của funding events được cung cấp; margin là **fixed entry margin budget** đang dùng, không phải mô hình margin thực tế của sàn. Khi budget bằng vốn ban đầu, phí entry có thể khiến equity thấp hơn budget ngay ở entry dù leverage 1×.

Target được notebook tính sẵn nên engine không thể kiểm tra notebook có dùng dữ liệu tương lai hay không. Chưa có callback state/actual fill cho feature logic: **chưa hỗ trợ entry-dependent TP/SL**. Không được dùng Close giả làm entry hoặc đọc next Open để quyết định target hiện tại. Thay đổi target ở Close[t] chỉ có thể ảnh hưởng fill từ observed bar kế tiếp.

Tất cả outputs ở RAM; chỉ khi researcher tự gọi save mới có artifacts. Không thêm Numba; correctness được kiểm tra bằng ví dụ tính tay trước khi benchmark dữ liệu thật.
