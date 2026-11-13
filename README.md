# Crypto research workspace

Dự án phục vụ thu thập dữ liệu, nghiên cứu alpha định hướng giá, backtest và phát triển hệ thống giao dịch tự động. Đã có dữ liệu OHLCV futures 1 phút và module backtest directional một asset.

Backtest một asset: [hướng dẫn bt_crypto](docs/bt_crypto.md), module [bt_crypto.py](bt_crypto.py) và [notebook alpha mẫu BTC](research/btc_alpha38_vwpd_5m.ipynb).

Grid search nhanh: [hướng dẫn scanner](docs/fast_grid.md), cài bằng `.\.venv\Scripts\python.exe -m pip install -r requirements-grid.txt`.

## Cài đặt nhanh sau khi clone

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

Mở `research/btc_alpha38_vwpd_5m.ipynb` trong VS Code, chọn kernel `.venv\Scripts\python.exe` và chạy hai cell lần lượt. Cell đầu chạy backtest, cell sau quét grid. Nến đã xử lý của 5 cặp đi kèm trong `data/processed/binance_um/1m/`; việc clone cần tải khoảng 1 GB. Grid results được lưu local, chưa bao gồm funding.

## Bộ dữ liệu

- Sàn: **Binance USDⓈ-M Futures**, hợp đồng perpetual ký quỹ USDT.
- Cặp: `BTCUSDT`, `ETHUSDT`, `XRPUSDT`, `BNBUSDT`, `DOGEUSDT`.
- Khung nến: `1m`, giá giao dịch thực tế (trade klines).
- Bắt đầu: **2021-01-01 00:00:00 UTC** (07:00 giờ Việt Nam).
- Kết thúc: snapshot local được tải ngày 04/11/2026. Khi chạy downloader, mốc cụ thể nằm trong `data/reports/quality_latest.json`.
- Timestamp lưu theo UTC. Tham số `--end` là cận trên không bao gồm; `--start` là cận dưới có bao gồm.
- `volume` tính theo đồng cơ sở; `quote_volume` tính theo USDT.

Nguồn chính thức: [Binance Public Data](https://github.com/binance/binance-public-data), [kho lưu trữ](https://data.binance.vision/), API `https://fapi.binance.com/fapi/v1/klines`. Không cần API key để tải dữ liệu công khai này.

## Môi trường riêng

Python **3.13**, môi trường ảo `.venv` ngay trong dự án. Các thư viện được cài riêng trong môi trường này. Bản cài chính xác nằm ở `requirements.lock.txt`.

Chạy từ `D:\Crypto_project` trong PowerShell:

```powershell
# Dùng trực tiếp Python của dự án, không cần activate
.\.venv\Scripts\python.exe --version

# Tải tiếp/cập nhật tới nến đã đóng gần nhất
.\.venv\Scripts\python.exe scripts\download_ohlcv.py

# Hoặc chọn khoảng UTC, end exclusive
.\.venv\Scripts\python.exe scripts\download_ohlcv.py --start 2021-01-01 --end 2026-01-01

# Kiểm tra logic phát hiện dữ liệu lỗi
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Đọc lại toàn bộ Parquet, kiểm tra độc lập đủ từng phút và OHLCV
.\.venv\Scripts\python.exe scripts\verify_data.py
```

Tạo lại môi trường khi cần:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

## Cấu trúc

```text
.venv/                                 Môi trường Python riêng
scripts/download_ohlcv.py               Tải lịch sử và cập nhật REST
scripts/verify_data.py                  Kiểm tra toàn bộ Parquet offline
scripts/update_data.ps1                 Lệnh PowerShell tiện dụng
research/load_data.py                  Hàm đọc dữ liệu theo cặp/khoảng UTC
tests/                                 Kiểm tra logic chất lượng dữ liệu
data/raw/binance_um/<symbol>/           ZIP nguồn và CHECKSUM SHA-256
data/processed/binance_um/1m/<symbol>/   Parquet Zstandard, mỗi tháng một file
data/reports/quality_latest.json        Báo cáo từng tháng và khoảng thiếu nến
data/reports/summary.csv                Tổng hợp từng cặp
data/reports/verification_latest.json   Kết quả đọc kiểm tra toàn bộ dữ liệu đã lưu
data/metadata/binance_exchange_info.json Thông tin 5 hợp đồng từ sàn lúc khởi tạo
logs/download.log                      Nhật ký tải
```

File Parquet có các cột:

| Cột | Ý nghĩa |
| --- | --- |
| `timestamp` | Thời gian mở nến, datetime UTC |
| `open_time` | Thời gian mở nến, Unix milliseconds, int64 |
| `open`, `high`, `low`, `close` | Giá OHLC, float64 |
| `volume` | Khối lượng đồng cơ sở, float64 |
| `close_time` | Thời gian đóng nến, Unix milliseconds, int64 |
| `quote_volume` | Khối lượng USDT, float64 |
| `trades` | Số giao dịch, int64 |
| `taker_buy_base_volume` | Khối lượng mua chủ động, đồng cơ sở |
| `taker_buy_quote_volume` | Khối lượng mua chủ động, USDT |

## Đọc dữ liệu cho research

Chạy Python từ thư mục gốc dự án:

```python
from research.load_data import load_ohlcv

btc = load_ohlcv('BTCUSDT', start='2025-01-01', end='2025-02-01')
print(btc.head())
print(btc.shape)
```

Hoặc đọc một tháng trực tiếp:

```python
import pandas as pd

df = pd.read_parquet('data/processed/binance_um/1m/BTCUSDT/2021-01.parquet')
```

## Tải và kiểm tra chất lượng

Downloader ưu tiên archive tháng, dùng archive ngày khi tháng chưa phát hành, bổ sung phần thiếu bằng REST. Mỗi ZIP được kiểm tra SHA-256 với file CHECKSUM của Binance và giữ lại để đối chiếu. Parquet ghi qua file tạm rồi đổi tên để có thể chạy lại sau gián đoạn.

Trước khi lưu, kiểm tra timestamp trùng, căn phút, thời gian đóng nến, giá OHLC hợp lệ, giá trị null và volume âm. Kiểm tra đủ từng phút trong khoảng yêu cầu, kể cả hai biên và các tháng. Nếu API vẫn không có dữ liệu, giữ khoảng trống và báo cụ thể, không tự nội suy hoặc chèn nến giả. Lệnh trả mã lỗi khi có phân vùng tải thất bại hoặc thiếu phút.

Chạy lại sẽ đọc Parquet đã có và tải các phút còn thiếu. Khi yêu cầu một khoảng nhỏ hơn, dữ liệu đã lưu ngoài khoảng đó được giữ lại. File ZIP cache được giữ nguyên; muốn tải lại một archive đã được Binance sửa, cần xóa đúng file ZIP, CHECKSUM và Parquet tháng tương ứng rồi chạy lại.

Dữ liệu Parquet đã xử lý được chia sẻ qua Git; ZIP gốc, reports tạo local và kết quả grid không đưa vào Git. Phạm vi dữ liệu hiện tại là OHLCV; funding, order book và mô phỏng khớp lệnh thực tế cần được bổ sung riêng.
