import numpy as np
import pandas as pd


class TechnicalIndicators(object):
    @staticmethod
    def adx(high, low, close, lookback=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(lookback).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1 / lookback).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1 / lookback).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
        adx_smooth = adx.ewm(alpha=1 / lookback).mean()

        return adx_smooth

    @staticmethod
    def stoch(high, low, close, lookback=14):
        high_lookback = high.rolling(lookback).max()
        low_lookback = low.rolling(lookback).min()
        slow_k = (close - low_lookback) * 100 / (high_lookback - low_lookback)
        slow_d = slow_k.rolling(3).mean()

        return slow_k, slow_d

    @staticmethod
    def chop(df, lookback=14):
        atr1 = TechnicalIndicators.atr(df['Mid_High'], df['Mid_Low'], df['Mid_Close'], lookback=1)
        high, low = df['Mid_High'], df['Mid_Low']

        chop = np.log10(
            atr1.rolling(lookback).sum() / (high.rolling(lookback).max() - low.rolling(lookback).min())) / np.log10(
            lookback)

        return chop

    @staticmethod
    def vo(volume, short_lookback=5, long_lookback=10):
        short_ema = pd.Series.ewm(volume, span=short_lookback).mean()
        long_ema = pd.Series.ewm(volume, span=long_lookback).mean()

        volume_oscillator = (short_ema - long_ema) / long_ema

        return volume_oscillator

    @staticmethod
    def williams_r(highs, lows, closes, length=21, ema_length=15):
        highest_highs = highs.rolling(window=length).max()
        lowest_lows = lows.rolling(window=length).min()

        willy = 100 * (closes - highest_highs) / (highest_highs - lowest_lows)
        willy_ema = pd.Series.ewm(willy, span=ema_length).mean()

        return willy, willy_ema

    @staticmethod
    def atr(high, low, close, lookback=14):
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        return true_range.rolling(lookback).mean()

    @staticmethod
    def atr_bands(high, low, close, lookback=14, atr_multiplier=3):
        scaled_atr_vals = TechnicalIndicators.atr(high, low, close, lookback) * atr_multiplier
        lower_band = close - scaled_atr_vals
        upper_band = close + scaled_atr_vals

        return lower_band, upper_band

    @staticmethod
    def rsi(closes, periods=14):
        close_delta = closes.diff()

        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100 / (1 + rsi))

        return rsi

    @staticmethod
    def qqe_mod(closes, rsi_period=6, smoothing=5, qqe_factor=3, threshold=3, mult=0.35, sma_length=50):
        Rsi = TechnicalIndicators.rsi(closes, rsi_period)
        RsiMa = Rsi.ewm(span=smoothing).mean()
        AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
        Wilders_Period = rsi_period * 2 - 1
        MaAtrRsi = AtrRsi.ewm(span=Wilders_Period).mean()
        dar = MaAtrRsi.ewm(span=Wilders_Period).mean() * qqe_factor

        longband = pd.Series(0.0, index=Rsi.index)
        shortband = pd.Series(0.0, index=Rsi.index)
        trend = pd.Series(0, index=Rsi.index)

        DeltaFastAtrRsi = dar
        RSIndex = RsiMa
        newshortband = RSIndex + DeltaFastAtrRsi
        newlongband = RSIndex - DeltaFastAtrRsi
        longband = pd.Series(np.where((RSIndex.shift(1) > longband.shift(1)) & (RSIndex > longband.shift(1)),
                                      np.maximum(longband.shift(1), newlongband), newlongband))
        shortband = pd.Series(np.where((RSIndex.shift(1) < shortband.shift(1)) & (RSIndex < shortband.shift(1)),
                                       np.minimum(shortband.shift(1), newshortband), newshortband))
        cross_1 = (longband.shift(1) < RSIndex) & (longband > RSIndex)
        cross_2 = (RSIndex > shortband.shift(1)) & (RSIndex.shift(1) < shortband)
        trend = np.where(cross_2, 1, np.where(cross_1, -1, trend.shift(1).fillna(1)))
        FastAtrRsiTL = pd.Series(np.where(trend == 1, longband, shortband))

        basis = (FastAtrRsiTL - 50).rolling(sma_length).mean()
        dev = mult * (FastAtrRsiTL - 50).rolling(sma_length).std()
        upper = basis + dev
        lower = basis - dev

        Greenbar1 = RsiMa - 50 > threshold
        Greenbar2 = RsiMa - 50 > upper

        Redbar1 = RsiMa - 50 < 0 - threshold
        Redbar2 = RsiMa - 50 < lower

        Greenbar = Greenbar1 & Greenbar2
        Redbar = Redbar1 & Redbar2

        return Greenbar, Redbar, RsiMa - 50

    @staticmethod
    def supertrend(barsdata, atr_len=10, mult=3):
        curr_atr = TechnicalIndicators.atr(barsdata['Mid_High'], barsdata['Mid_Low'], barsdata['Mid_Close'],
                                           lookback=atr_len)
        highs, lows = barsdata['Mid_High'], barsdata['Mid_Low']
        hl2 = (highs + lows) / 2
        final_upperband = hl2 + mult * curr_atr
        final_lowerband = hl2 - mult * curr_atr

        # initialize Supertrend column to True
        supertrend = [True] * len(barsdata)

        close = barsdata['Mid_Close']

        for i in range(1, len(barsdata.index)):
            curr, prev = i, i - 1

            # if current close price crosses above upperband
            if close[curr] > final_upperband[prev]:
                supertrend[curr] = True

            # if current close price crosses below lowerband
            elif close[curr] < final_lowerband[prev]:
                supertrend[curr] = False

            # else, the trend continues
            else:
                supertrend[curr] = supertrend[prev]

                # adjustment to the final bands
                if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                    final_lowerband[curr] = final_lowerband[prev]

                if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                    final_upperband[curr] = final_upperband[prev]

        return supertrend, final_upperband, final_lowerband

    @staticmethod
    def squeeze(barsdata, length=20, length_kc=20, mult=1.5):
        # Bollinger bands
        m_avg = barsdata['Mid_Close'].rolling(window=length).mean()
        m_std = barsdata['Mid_Close'].rolling(window=length).std(ddof=0)
        upper_bb = m_avg + mult * m_std
        lower_bb = m_avg - mult * m_std

        # Keltner channel
        tr0 = abs(barsdata['Mid_High'] - barsdata['Mid_Low'])
        tr1 = abs(barsdata['Mid_High'] - barsdata['Mid_Close'].shift())
        tr2 = abs(barsdata['Mid_Low'] - barsdata['Mid_Close'].shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        range_ma = tr.rolling(window=length_kc).mean()
        upper_kc = m_avg + range_ma * mult
        lower_kc = m_avg - range_ma * mult

        # Squeeze
        squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)

        return squeeze_on
