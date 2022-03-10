import pandas as pd


def resample_ohlc(df, period):
    output = pd.DataFrame()
    period = period.replace("m", "T")

    if "open" in df.columns:
        output["open"] = df.open.resample(period).first()
        output["high"] = df.high.resample(period).max()
        output["low"] = df.low.resample(period).min()
        output["close"] = df.close.resample(period).last()
    else:
        output = df.resample(period).ffill()

    return output
