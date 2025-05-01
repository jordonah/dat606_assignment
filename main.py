from yahoofinance import StockPredictor


if __name__ == '__main__':
    sp = StockPredictor("2015-01-01","2025-01-17")
    sp.convert_to_dates(75)
    sp.download_data(ticker='AAPL')
    sp.preprocess_data()
    sp.scale_data()


    # main.py --predict "2015-01-01" "2025-01-01" 80 "AAPL"