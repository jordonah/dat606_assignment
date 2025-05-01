from yahoofinance import StockPredictor
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stock Price Prediction")  
    parser.add_argument('--predict', nargs=4, help='For predicting stock price in a given ticker for a date range.')
    args = parser.parse_args()
    if args.predict is not None and len(args.predict) == 4:
        sp = StockPredictor(args.predict[0], args.predict[1])
        sp.convert_to_dates(args.predict[2])
        sp.download_data(ticker=args.predict[3])
        sp.preprocess_data()
        sp.scale_data()


    # main.py --predict "2015-01-01" "2024-06-01" 80 "AAPL"