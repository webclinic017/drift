import pandas as pd


def resample_ohlc(df, period):
    output = pd.DataFrame()
    output["open"] = df.open.resample(period).first()
    output["high"] = df.high.resample(period).max()
    output["low"] = df.low.resample(period).min()
    output["close"] = df.close.resample(period).last()
    return output


def upsample(df, period):
    output = pd.DataFrame()
    period = period.replace("m", "T")
    output = df.resample(period).ffill()

    return output
