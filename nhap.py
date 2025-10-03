# from FiinQuantX import FiinSession
from datetime import datetime, timedelta
# client = FiinSession(username="DSTC_37@fiinquant.vn", password="Fiinquant0606").login()
# hose = list(client.TickerList(tickers="VNINDEX"))
# hnx = list(client.TickerList(tickers="HNXINDEX"))
# upcom = list(client.TickerList(tickers="UPCOMINDEX"))
# all_tickers = hose + hnx + upcom
# print(len(all_tickers))
# help(FiinSession.TickerList)
from FiinQuantX import FiinSession, RealTimeData
def on_event(data: RealTimeData):
    """
    Hàm callback được gọi mỗi khi có dữ liệu mới từ FiinQuantX.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ticker = data.Ticker
    price = data.Close
    
    print(f"Thời gian: {timestamp}, Cổ phiếu: {ticker}, Giá: {price}")

client = FiinSession(username="DSTC_37@fiinquant.vn", password="Fiinquant0606").login()
client.Trading_Data_Stream(tickers="VCB", callback=on_event)
