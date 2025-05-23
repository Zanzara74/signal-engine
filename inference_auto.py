#!/usr/bin/env python3
"""
inference_auto.py

Automatically processes ALL scenarios:
  • Finds every <name>_history_events.csv + <name>_model.pkl
  • Predicts BUY signals, fetches entry_price + entry_time
  • Saves signals_<name>_<YYYY-MM-DD>.csv per scenario
  • Uploads each to Drive under DRIVE_FOLDER_ID
  • Sends each as a Telegram document with a caption of the signals
"""

import os
import glob
from datetime import datetime, timedelta
import pandas as pd
import joblib
import requests
import yfinance as yf
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ── CONFIG ────────────────────────────────────────────────────────────
SERVICE_ACCOUNT_JSON = os.environ['SERVICE_ACCOUNT_JSON']
DRIVE_FOLDER_ID      = os.environ['DRIVE_FOLDER_ID']
TELEGRAM_BOT_TOKEN   = os.environ['TELEGRAM_BOT_TOKEN']
TELEGRAM_CHAT_ID     = os.environ['TELEGRAM_CHAT_ID']
SERVICE_JSON         = 'service_account.json'
OUTPUT_BASE          = 'daily_signals'


# ── DRIVE SETUP ───────────────────────────────────────────────────────
def init_drive():
    with open(SERVICE_JSON, 'w') as f:
        f.write(SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_JSON,
        scopes=['https://www.googleapis.com/auth/drive.file']
    )
    return build('drive', 'v3', credentials=creds)


# ── FEATURE HELPERS ────────────────────────────────────────────────────
def compute_pct_return(symbol, ts):
    try:
        hist = yf.Ticker(symbol).history(
            start=ts.date() - timedelta(days=2),
            end=ts.date() + timedelta(days=1)
        )['Close']
        if len(hist) >= 2:
            return (hist.iloc[-1] - hist.iloc[-2]) / hist.iloc[-2]
    except Exception:
        pass
    return 0.0


def fetch_price(symbol, ts):
    try:
        hist = yf.Ticker(symbol).history(
            start=ts.date() - timedelta(days=2),
            end=ts.date() + timedelta(days=1)
        )['Close']
        return float(hist.iloc[-1])
    except Exception:
        return None


# ── PROCESS ONE SCENARIO ──────────────────────────────────────────────
def process_scenario(name, csv_path, model_path, drive_svc):
    df = pd.read_csv(csv_path, header=0, names=['DateRaw','Ticker'])
    df = df.rename(columns={'Ticker': 'symbol'})
    df['event_timestamp'] = pd.to_datetime(
        df['DateRaw'], format='%Y-%m-%d', errors='coerce', utc=True
    )

    model = joblib.load(model_path)

    df['entry_time'] = df['event_timestamp']
    df['pct_return'] = df.apply(
        lambda r: compute_pct_return(r['symbol'], r['entry_time']),
        axis=1
    )
    X = df[['pct_return']]
    df['signal'] = model.predict(X)

    buys = df[df['signal'] == 1].copy()
    if buys.empty:
        return None

    buys['entry_price'] = buys.apply(
        lambda r: fetch_price(r['symbol'], r['entry_time']), axis=1
    )
    buys['entry_time'] = buys['entry_time'].dt.strftime('%d/%m/%y %H:%M')

    date_str = datetime.utcnow().strftime('%Y-%m-%d')
    folder   = os.path.join(OUTPUT_BASE, name)
    os.makedirs(folder, exist_ok=True)
    out_csv  = os.path.join(folder, f"{name}_signals_{date_str}.csv")
    buys[['symbol','entry_price','entry_time']].to_csv(out_csv, index=False)

    meta  = {'name': os.path.basename(out_csv), 'parents': [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(out_csv, mimetype='text/csv')
    drive_svc.files().create(body=meta, media_body=media).execute()

    return out_csv, buys


# ── TELEGRAM NOTIFICATION ─────────────────────────────────────────────
def send_telegram(file_path, buys, scenario_name):
    date_hdr = datetime.utcnow().strftime('%d/%m/%y')
    if not buys.empty:
        # properly closed f-string with newline
        caption = f"📈 {scenario_name.upper()} signals for {date_hdr} (GMT):\n"
        for _, r in buys.iterrows():
            d, t = r['entry_time'].split(' ')
            p    = r['entry_price']
            caption += f"• {r['symbol']} @ {p if p is not None else 'N/A'} on {d} {t} GMT\n"

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        with open(file_path, 'rb') as f:
            requests.post(
                url,
                params={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption},
                files={'document': f}
            )
    else:
        # fallback (rare)
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={'chat_id': TELEGRAM_CHAT_ID,
                  'text': f"✅ No BUY signals for {scenario_name} today."}
        )


# ── MAIN ENTRYPOINT ───────────────────────────────────────────────────
def main():
    drive = init_drive()
    any_signals = False

    for csv_path in glob.glob("*_history_events.csv"):
        name       = os.path.basename(csv_path).replace("_history_events.csv", "")
        model_path = f"{name}_model.pkl"
        if not os.path.exists(model_path):
            print(f"⚠️  Skipping '{name}': missing model.")
            continue

        result = process_scenario(name, csv_path, model_path, drive)
        if result:
            file_path, buys = result
            send_telegram(file_path, buys, name)
            any_signals = True
        else:
            print(f"ℹ️  No BUY signals for '{name}' today.")

    if not any_signals:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={'chat_id': TELEGRAM_CHAT_ID,
                  'text': "✅ No BUY signals in any scenario today."}
        )


if __name__ == "__main__":
    main()
