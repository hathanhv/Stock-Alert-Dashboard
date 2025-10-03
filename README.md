# 📈 Stock Alert Dashboard

Hệ thống dashboard cảnh báo giao dịch chứng khoán realtime sử dụng Streamlit và Machine Learning để dự đoán giá cổ phiếu và sinh tín hiệu BUY/SELL/HOLD.

## 🎯 Tính năng chính

- **📊 Monitoring Realtime**: Theo dõi giá cổ phiếu realtime với FiinQuantX
- **🤖 AI Trading Signals**: Dự đoán xác suất tăng giá 10 ngày bằng ML model
- **📈 Technical Analysis**: Chỉ số kỹ thuật đầy đủ (RSI, MACD, Bollinger Bands, EMA, SMA...)
- **💰 Fundamental Analysis**: Tích hợp báo cáo tài chính (P/E, P/B, ROE, EPS, Growth rates)
- **🔄 Backtesting**: Đánh giá chiến lược trên dữ liệu lịch sử với metrics chi tiết
- **⚙️ Model Management**: Tái train model định kỳ và quản lý cấu hình

## 🏗️ Cấu trúc dự án

```
stock_alert_dashboard/
│
├── data/
│   ├── historical/          # Dữ liệu lịch sử OHLCV
│   └── realtime/            # Cache dữ liệu realtime
│
├── models/
│   └── checkpoint.pkl       # Model đã train
│
├── utils/
│   ├── fetch_data.py        # Lấy dữ liệu lịch sử, realtime, báo cáo tài chính
│   ├── preprocess.py        # Merge và chuẩn hóa dữ liệu
│   ├── features.py          # Tính chỉ số kỹ thuật
│   ├── signals.py           # Sinh tín hiệu BUY/SELL/HOLD
│   └── backtest.py          # Backtest và đánh giá chiến lược
│
├── pages/
│   ├── dashboard.py         # Dashboard chính realtime
│   ├── backtest_page.py     # Trang backtest
│   └── settings.py          # Cấu hình và quản lý model
│
├── app.py                   # Entrypoint Streamlit
├── config.py                # Cấu hình hệ thống
├── env_example.txt          # Template file .env
├── requirements.txt         # Dependencies
└── README.md               # Hướng dẫn này
```

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd stock_alert_dashboard
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Cấu hình credentials
```bash
# Copy file template
cp env_example.txt .env

# Chỉnh sửa file .env với thông tin FiinQuantX của bạn
USERNAME1=your_username_here
PASSWORD1=your_password_here
```

### 5. Chạy ứng dụng
```bash
streamlit run app.py
```

## 📖 Hướng dẫn sử dụng

### 🏠 Trang Home
- Tổng quan hệ thống và trạng thái
- Kiểm tra cấu hình và model
- Hướng dẫn setup nhanh

### 📈 Dashboard (Trang chính)
- **Monitoring Realtime**: Theo dõi giá cổ phiếu theo thời gian thực
- **Trading Signals**: Hiển thị tín hiệu BUY/SELL/HOLD với confidence score
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Fundamental Ratios**: P/E, P/B, ROE, EPS, Revenue/Profit Growth
- **Auto Refresh**: Tự động cập nhật dữ liệu theo interval

**Cách sử dụng:**
1. Nhập ticker cổ phiếu (VD: VCB, VIC, HPG)
2. Điều chỉnh thresholds cho BUY/SELL signals
3. Bật Auto Refresh để theo dõi realtime
4. Xem charts, indicators và signals

### 📊 Backtest
- **Strategy Testing**: Test chiến lược trên dữ liệu lịch sử
- **Performance Metrics**: ROI, Sharpe ratio, Max drawdown, Win rate
- **Trade Analysis**: Chi tiết các giao dịch và commission
- **Charts**: Equity curve, returns distribution, drawdown

**Cách sử dụng:**
1. Chọn ticker và khoảng thời gian backtest
2. Điều chỉnh parameters (capital, commission, thresholds)
3. Click "Run Backtest" để chạy
4. Xem kết quả và download reports

### ⚙️ Settings
- **Configuration**: Điều chỉnh thresholds và parameters
- **Model Management**: Train model mới, backup/restore
- **Data Management**: Clear cache, refresh data
- **System Status**: Thông tin hệ thống và performance

**Model Training:**
1. Vào tab "Model Management"
2. Nhập danh sách tickers để train
3. Chọn số ngày dữ liệu lịch sử
4. Click "Train New Model"
5. Xem metrics và performance

## 🔧 Cấu hình nâng cao

### Signal Thresholds
```python
# Trong config.py hoặc Settings page
BUY_THRESHOLD = 0.7    # Xác suất >= 70% → BUY
SELL_THRESHOLD = 0.3   # Xác suất <= 30% → SELL
# 30% < Xác suất < 70% → HOLD
```

### Technical Indicators
```python
RSI_PERIOD = 14
EMA_PERIODS = [5, 10, 20, 50]
SMA_PERIODS = [20, 50, 100, 200]
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
```

### Backtest Parameters
```python
INITIAL_CAPITAL = 1000000  # 1M VND
COMMISSION_RATE = 0.0015   # 0.15%
PREDICTION_DAYS = 10       # Dự đoán 10 ngày
```

## 📊 Data Pipeline

### 1. Data Fetching
- **Historical**: FiinQuantX.Fetch_Trading_Data
- **Realtime**: FiinQuantX.Trading_Data_Stream
- **Fundamental**: FiinQuantX.FundamentalAnalysis.get_ratios

### 2. Feature Engineering
- **Technical**: RSI, MACD, Bollinger, EMA, SMA, Volume indicators
- **Fundamental**: P/E, P/B, ROE, EPS, Growth rates
- **Lag Features**: Historical values của key indicators
- **Rolling Features**: Moving averages và statistics

### 3. Model Prediction
- **Input**: Combined technical + fundamental features
- **Output**: Probability of price increase in 10 days
- **Signal**: BUY/SELL/HOLD based on thresholds + technical filters

### 4. Signal Generation
- **Model Output** → Probability
- **Technical Filters** → RSI overbought/oversold, MACD crossover
- **Final Signal** → BUY/SELL/HOLD with confidence

## 🧠 Model Architecture

### Training Data
- **Features**: 50+ technical + fundamental indicators
- **Target**: Binary (1 if price increases >2% in 10 days, 0 otherwise)
- **Timeframe**: Rolling window training với latest data

### Model Type
- **Algorithm**: Random Forest Classifier
- **Parameters**: n_estimators=100, max_depth=10, class_weight='balanced'
- **Validation**: 80/20 train/test split

### Retraining Strategy
- **Frequency**: Monthly hoặc khi có đủ data mới
- **Method**: Full retrain hoặc incremental learning
- **Validation**: Cross-validation với time series split

## 📈 Performance Metrics

### Trading Metrics
- **Total Return**: Lợi nhuận tổng thể
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Mức sụt giảm tối đa
- **Win Rate**: Tỷ lệ giao dịch thắng
- **Average Trade Return**: Lợi nhuận trung bình mỗi giao dịch

### Model Metrics
- **Accuracy**: Độ chính xác dự đoán
- **Precision/Recall**: Cho từng class
- **Confusion Matrix**: Chi tiết predictions
- **Feature Importance**: Các features quan trọng nhất

## 🔍 Troubleshooting

### Common Issues

**1. "No model found"**
```bash
# Solution: Train a new model
# Vào Settings → Model Management → Train New Model
```

**2. "FiinQuantX connection failed"**
```bash
# Solution: Check credentials
# Kiểm tra file .env có đúng username/password không
```

**3. "No data available for ticker"**
```bash
# Solution: Check ticker symbol
# Đảm bảo ticker đúng format (VD: VCB, VIC, HPG)
```

**4. "Import errors"**
```bash
# Solution: Install missing packages
pip install -r requirements.txt
```

### Log Files
- **Signals**: `data/signals_log.csv`
- **Application**: Console output và Streamlit logs
- **Model**: `models/checkpoint.pkl` và `models/feature_names.pkl`

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 8501
```

### Production Deployment
```bash
# Using Streamlit Cloud hoặc Docker
# Đảm bảo set environment variables
export USERNAME1="your_username"
export PASSWORD1="your_password"
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📝 API Reference

### DataFetcher
```python
fetcher = DataFetcher()
# Historical data
data = fetcher.fetch_historical_data("VCB", days_back=365)
# Realtime data
realtime = fetcher.fetch_realtime_data(["VCB"], callback=on_event)
# Financial ratios
ratios = fetcher.fetch_financial_ratios(["VCB"])
```

### SignalGenerator
```python
generator = SignalGenerator()
# Predict probability
prob = generator.predict_probability(features)
# Generate signal
signal = generator.generate_signal(prob, price, indicators)
```

### Backtester
```python
backtester = Backtester(initial_capital=1000000)
# Run backtest
results = backtester.run_backtest(data, signals, start_date, end_date)
# Get metrics
metrics = backtester.calculate_performance_metrics(data)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This tool is for educational and research purposes only. It is not intended as financial advice.**

- Không phải lời khuyên đầu tư
- Rủi ro cao, có thể mất vốn
- Nên test kỹ trên paper trading trước
- Luôn diversify portfolio
- Consult với financial advisor

## 📞 Support

- **Issues**: Tạo issue trên GitHub
- **Documentation**: Đọc README và code comments
- **Community**: Thảo luận trên GitHub Discussions

---

**📈 Happy Trading! 🚀**
