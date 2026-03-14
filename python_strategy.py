import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
import datetime
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def calculate_supertrend(df, atr_length, multiplier):
    """
    Custom SuperTrend calculation mimicking Pine Script logic.
    Provides up_trend and down_trend based on Close, High, Low.
    """
    if multiplier > 0:
        atr = df.ta.atr(length=atr_length)
    else:
        atr = pd.Series(0.0, index=df.index)
        
    up_lev = df['Low'] - multiplier * atr
    dn_lev = df['High'] + multiplier * atr
    
    up_trend = np.zeros(len(df))
    down_trend = np.zeros(len(df))
    trend = np.ones(len(df))
    
    close = df['Close'].values
    up_lev_vals = up_lev.fillna(0).values
    dn_lev_vals = dn_lev.fillna(0).values
    
    for i in range(1, len(df)): # Basic loop for stateful supertrend logic
        # Up trend logic
        if close[i-1] > up_trend[i-1]:
            up_trend[i] = max(up_lev_vals[i], up_trend[i-1])
        else:
            up_trend[i] = up_lev_vals[i]
            
        # Down trend logic
        if close[i-1] < down_trend[i-1]:
            down_trend[i] = min(dn_lev_vals[i], down_trend[i-1])
        else:
            down_trend[i] = dn_lev_vals[i]
            
        # Trend direction logic
        if close[i] > down_trend[i-1]:
            trend[i] = 1
        elif close[i] < up_trend[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
    st_line = np.where(trend == 1, up_trend, down_trend)
    return pd.Series(st_line, index=df.index)


def calculate_indicators(df):
    """
    Calculate the underlying indicators required for strategy signals.
    Expects a DataFrame with OHLCV structure and Datetime index.
    """
    df = df.copy()
    
    # Shared variables
    df['hlc3'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['ohlc4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # ----------------------------------------------------
    # 1. Dollar Cost Average Strategy (DCA)
    # ----------------------------------------------------
    # xmf (Money Flow Approximation)
    hlc3_diff = df['hlc3'].diff()
    upper_s_vol = np.where(hlc3_diff <= 0, 0, df['hlc3']) * df['Volume']
    lower_s_vol = np.where(hlc3_diff >= 0, 0, df['hlc3']) * df['Volume']
    
    upper_s = pd.Series(upper_s_vol, index=df.index).rolling(14).sum()
    lower_s = pd.Series(lower_s_vol, index=df.index).rolling(14).sum()
    
    # Prevent division by zero
    lower_s = lower_s.replace(0, np.nan)
    xmf = 100.0 - (100.0 / (1.0 + upper_s / lower_s))
    
    # BollOsc (Bollinger Oscillator)
    basis = df['ohlc4'].rolling(25).mean()
    dev = 40 * df['ohlc4'].rolling(25).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    OB1 = (upper + lower) / 2.0
    OB2 = (upper - lower).replace(0, np.nan)
    BollOsc = (df['ohlc4'] - OB1) / OB2 * 100
    
    # xrsi
    xrsi = ta.rsi(df['ohlc4'], length=14)
    
    # stoc
    ll3 = df['Low'].rolling(21).min()
    hh = df['High'].rolling(21).max()
    k = 100 * (df['ohlc4'] - ll3) / (hh - ll3)
    stoc = k.rolling(3).mean()
    
    # trend23
    trend23 = (xrsi + xmf + BollOsc + stoc / 3) / 2
    
    # reg_trend23 (Linear Regression on trend23)
    length = 10
    x = pd.Series(np.arange(len(df)), index=df.index)
    y = trend23.fillna(0)
    
    x_ema = ta.ema(x, length=length)
    y_ema = ta.ema(y, length=length)
    
    mx = x.rolling(length).std(ddof=0)
    my = y.rolling(length).std(ddof=0)
    c23 = x.rolling(length).corr(y)
    
    slope = c23 * (my / mx)
    inter = y_ema - slope * x_ema
    reg_trend23 = x * slope + inter
    
    # buy_4 logic (DCA input/signal)
    min_level = 36
    crossunder = (reg_trend23.shift(1) >= trend23.shift(1)) & (reg_trend23 < trend23)
    df['dca_signal'] = crossunder & (reg_trend23 <= min_level) & (trend23 <= min_level)
    
    
    # ----------------------------------------------------
    # 2. Bull vs Bear Strategy (Higher Timeframe '2D')
    # ----------------------------------------------------
    # Note: Pine script uses "barmerge.lookahead_on" which calculates using future HTF closed data
    # creating lookahead bias. We use standard `.ffill()` over resampling to maintain 
    # realistic backtesting integrity, avoiding lookahead. 
    df_2d = df.resample('2D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    st_line_2d = calculate_supertrend(df_2d, atr_length=500, multiplier=0)
    # Map back to original timeframe safely
    st_line_mapped_2d = st_line_2d.reindex(df.index).ffill()
    
    df['buy44'] = (df['Close'].shift(1) <= st_line_mapped_2d.shift(1)) & (df['Close'] > st_line_mapped_2d)
    df['sell44'] = (df['Close'].shift(1) >= st_line_mapped_2d.shift(1)) & (df['Close'] < st_line_mapped_2d)
    
    
    # ----------------------------------------------------
    # 3. The Gods Trend (Higher Timeframe '1D')
    # ----------------------------------------------------
    df_1d = df.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    st_line_1d = calculate_supertrend(df_1d, atr_length=4, multiplier=0.05)
    st_line_mapped_1d = st_line_1d.reindex(df.index).ffill()
    
    df['thegodsbuy'] = (df['Close'].shift(1) <= st_line_mapped_1d.shift(1)) & (df['Close'] > st_line_mapped_1d)
    df['thegodssell'] = (df['Close'].shift(1) >= st_line_mapped_1d.shift(1)) & (df['Close'] < st_line_mapped_1d)
    

    # ----------------------------------------------------
    # Composite Signals Creation
    # ----------------------------------------------------
    df['long_entry'] = df['buy44'] | df['thegodsbuy']
    df['short_entry'] = df['sell44'] | df['thegodssell']
    
    return df

def apply_strategy_exits(df):
    """
    A basic simulation function mimicking the Pine Script Takeprofit (TP) 
    and Stoploss (SL) mechanics tracking trade entries.
    Recommended: use frameworks like VectorBT or Backtrader for robust backtests.
    """
    # Pine Script percentage configuration
    stopPerHR1 = 0.10
    takePerHR3 = 0.24 # Full max out logic for simplicity in loop
    
    df['position_price'] = np.nan
    df['active_position'] = 0 # 1 out Long, -1 short, 0 flat
    
    pos_type = 0
    entry_price = 0.0
    signals = []
    
    # Basic illustrative loop for state management
    for i in range(len(df)):
        if pos_type == 0:
            if df['long_entry'].iloc[i]:
                pos_type = 1
                entry_price = df['Close'].iloc[i]
            elif df['short_entry'].iloc[i]:
                pos_type = -1
                entry_price = df['Close'].iloc[i]
        elif pos_type == 1:
            sl_price = entry_price * (1 - stopPerHR1)
            tp_price = entry_price * (1 + takePerHR3) 
            if df['Low'].iloc[i] < sl_price or df['High'].iloc[i] > tp_price:
                pos_type = 0 # SL or full TP hit
        elif pos_type == -1:
            sl_price = entry_price * (1 + stopPerHR1)
            tp_price = entry_price * (1 - takePerHR3)
            if df['High'].iloc[i] > sl_price or df['Low'].iloc[i] < tp_price:
                pos_type = 0
                
        signals.append({'pos_type': pos_type, 'entry_price': entry_price if pos_type != 0 else np.nan})
        
    res = pd.DataFrame(signals, index=df.index)
    df['active_position'] = res['pos_type']
    return df

def send_telegram_alert(bot_token, chat_id, symbol, signal_type, price, time_str):
    """
    Sends a formatted, professional alert message to Telegram.
    """
    if not bot_token or not chat_id:
        return
        
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    # Premium formatting
    emoji = "🟢" if signal_type.upper() == "LONG" else "🔴"
    
    message = (
        f"<b>{emoji} CRYPTO BOT ALERT</b>\n\n"
        f"<b>Pair:</b> #{symbol}\n"
        f"<b>Action:</b> {signal_type.upper()} Entry\n"
        f"<b>Price:</b> {price:,.4f}\n"
        f"<b>Time:</b> {time_str}\n\n"
        f"<i>Automated Signal Terminal</i>"
    )
    
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Telegram Alert Failed: {e}")

def fetch_delta_exchange_data(symbol="BTCUSD", resolution="1h", days_back=30):
    """
    Fetches real OHLCV data from Delta Exchange India public API.
    """
    url = "https://api.india.delta.exchange/v2/history/candles"
    
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=days_back)
    
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "start": int(start_time.timestamp()),
        "end": int(end_time.timestamp())
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    if data.get("success") and data.get("result"):
        df = pd.DataFrame(data["result"])
        
        # Map Delta Exchange dict output to Strategy expected format
        df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }, inplace=True)
        
        # Ensure DataFrame Index is Datetime and explicitly sorted
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        
        # Convert necessary columns to Float
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)
            
        return df
    else:
        raise Exception(f"Failed to fetch data from Delta Exchange API: {data}")

if __name__ == "__main__":
    # --- Execute a test run of the Strategy framework with REAL data ---
    symbols_to_test = ["BTCUSD", "ETHUSD", "SOLUSD"]
    
    # Load User Notification Setup from .env file
    TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
    TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")
    
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("Warning: Telegram credentials are not fully set in the .env file. Alerts will not be sent.")
    
    print("Bot is now running in the background. Press Ctrl+C to stop.")
    
    while True:
        print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting strategy scan...")
        for sym in symbols_to_test:
            try:
                print(f"\n=============================================")
                print(f"Fetching Real OHLCV data for {sym}...")
                print(f"=============================================")
                real_df = fetch_delta_exchange_data(symbol=sym, resolution="1h", days_back=30)
                print(f"Successfully fetched {len(real_df)} rows of recent 1h candles.")
                
                print("Calculating Custom Indicators...")
                result_df = calculate_indicators(real_df)
                
                print("Applying Exit Conditions (TP/SL)...")
                final_df = apply_strategy_exits(result_df)
                
                print(f"\n---------- {sym} Strategy Results Sample ----------")
                col_show = ['Close', 'long_entry', 'short_entry', 'dca_signal', 'active_position']
                print(final_df[col_show].tail(5))
                
                # --- Check For Live Alerts on the LATEST Candle ---
                latest_candle = final_df.iloc[-1]
                previous_candle = final_df.iloc[-2]
                
                # Identify fresh long/short crossings for the final bar downloaded
                if latest_candle['long_entry'] and not previous_candle['long_entry']:
                    time_str = str(latest_candle.name)
                    print(f"*** NEW LIVE ALERT: LONG on {sym} at {latest_candle['Close']} ***")
                    send_telegram_alert(TG_BOT_TOKEN, TG_CHAT_ID, sym, "LONG", latest_candle['Close'], time_str)
                    
                elif latest_candle['short_entry'] and not previous_candle['short_entry']:
                    time_str = str(latest_candle.name)
                    print(f"*** NEW LIVE ALERT: SHORT on {sym} at {latest_candle['Close']} ***")
                    send_telegram_alert(TG_BOT_TOKEN, TG_CHAT_ID, sym, "SHORT", latest_candle['Close'], time_str)
                
            except Exception as e:
                print(f"Error during execution for {sym}: {e}")
                
        print("\n---------------------------------------------")
        print("Python Strategy execution completed for all symbols.")
        print("Waiting 1 hour for the next hourly candle...")
        time.sleep(3600)
