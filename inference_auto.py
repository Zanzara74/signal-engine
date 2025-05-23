#!/usr/bin/env python3
"""
inference_auto.py

• Reads a two-col CSV: DateRaw,Ticker
• Renames to event_timestamp,symbol
• Computes pct_return + entry_price via yfinance
• Uploads CSV to Google Drive
• Sends a Telegram alert with symbol, price, date/time
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import joblib
import requests
import yfinance as yf
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# CONFIG
SERVICE_ACCOUNT_JSON = os.environ['SERVICE_ACCOUNT_JSON']
DRIVE_FOLDER_ID      = os.environ['DRIVE_FOLDER_ID']
TELEGRAM_BOT_TOKEN   = os.environ['TELEGRAM_BOT_TOKEN']
TELEGRAM_CHAT_ID     = os.environ['TELEGRAM_CHAT_ID']

EVENTS_FILE    = 'events_history.csv'
MODEL_FILE     = 'model.pkl'
OUTPUT_FOLDER  = 'daily_signals'
SERVICE_JSON   = 'service_account.json'

def init_drive():
    with open(SERVICE_JSON, 'w') as f:
        f.write(SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_JSON,
        scopes=['https://www.googleapis.com/auth/drive.file']
    )
    return build('drive', 'v3', credentials=creds)

def load_data():
    df = pd.read_csv(EVENTS_FILE, header=0, names=['DateRaw','Ticker'])
    df = df.rename(columns={'Ticker':'symbol'})
    df['event_timestamp'] = pd.to_datetime(
        df['DateRaw'], format='%Y-%m-%d', errors='coerce', utc=True
    )
    if df['event_timestamp'].isna().all():
        raise ValueError("Parsed no dates from DateRaw!")
    return df

def load_model():
    return joblib.load(MODEL_FILE)

def compute_pct_return(symbol, ts):
    try:
        hist = yf.Ticker(symbol).history(
            start=ts.date() - timedelta(days=2),
            end=ts.date() + timedelta(days=1)
        )['Close']
        if len(hist) >= 2:
            return (hist.iloc[-1] - hist.iloc[-2]) / hist.iloc[-2]
    except:
        pass
    return 0.0

def fetch_price(symbol, ts):
    try:
        hist = yf.Ticker(symbol).history(
            start=ts.date() - timedelta(days=2),
            end=ts.date() + timedelta(days=1)
        )['Close']
        return float(hist.iloc[-1])
    except:
        return None

def run_inference(df, model):
    df['entry_time'] = df['event_timestamp']
    df['pct_return'] = df.apply(
        lambda r: compute_pct_return(r['symbol'], r['entry_time']), axis=1
    )
    X = df[['pct_return']]
    df['signal'] = model.predict(X)
    buys = df[df['signal'] == 1].copy()
    buys['entry_price'] = buys.apply(
        lambda r: fetch_price(r['symbol'], r['entry_time']), axis=1
    )
    return buys[['symbol','entry_price','entry_time']]

def save_and_upload(drive_svc, df):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    date_str = datetime.utcnow().strftime('%Y-%m-%d')
    path = f"{OUTPUT_FOLDER}/signals_{date_str}.csv"
    df['entry_time'] = df['entry_time'].dt.strftime('%d/%m/%y %H:%M')
    df.to_csv(path, index=False)
    meta = {'name': os.path.basename(path), 'parents': [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(path, mimetype='text/csv')
    drive_svc.files().create(body=meta, media_body=media).execute()
    return path

def notify_telegram(df, csv_path):
    if df.empty:
        text = "✅ No BUY signals today."
    else:
        text = f"📈 BUY signals for {datetime.utcnow().strftime('%d/%m/%y')} (GMT):\n"
        for _, r in df.iterrows():
            d,t = r['entry_time'].split(' ')
            p = r['entry_price']
            text += f"• {r['symbol']} @ {p if p else 'N/A'} on {d} {t} GMT\n"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    with open(csv_path, 'rb') as f:
        files = {'document': f}
        params = {'chat_id': TELEGRAM_CHAT_ID, 'caption': text}
        requests.post(url, params=params, files=files)

def main():
    drive = init_drive()
    df = load_data()
    model = load_model()
    out = run_inference(df, model)
    csv = save_and_upload(drive, out)
    notify_telegram(out, csv)

if __name__ == "__main__":
    main()
