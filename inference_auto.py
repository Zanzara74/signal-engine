#!/usr/bin/env python3
"""
inference_auto.py

Automatically processes ALL scenarios:
  • Finds every <name>_history_events.csv + <name>_model.pkl
  • Predicts BUY signals, fetches entry_price + entry_time
  • Saves signals_<name>_<YYYY-MM-DD>.csv per scenario
  • Uploads each to Drive under DRIVE_FOLDER_ID
  • Sends each as a Telegram document with a caption of the signals
  • (Stub) Schedules exit reminders
"""

import os
import glob
import math
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
ENTRY_WINDOW_HOURS   = 8  # how many hours from event to suggest entry window


# ── DRIVE INITIALIZATION ──────────────────────────────────────────────
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


# ── EXIT REMINDER STUB ────────────────────────────────────────────────
def schedule_exit_reminder(symbol, entry_ts, scenario_name):
    """
    Stub: integrate with 'automations' tool or a scheduled job
    to notify you when to exit.
    E.g., for short holds reminder = entry_ts + 5 days.
    """
    # horizon_days = 5 if hold_cat == 'Short' else 20
    # reminder_time = entry_ts + timedelta(days=horizon_days)
    # ...call automations.create(...) here...
    pass


# ── PER-SCENARIO PROCESSING ────────────────────────────────────────────
def process_scenario(name, csv_path, model_path, drive_svc):
    # 1) Load raw events
    df = pd.read_csv(csv_path, header=0, names=['DateRaw','Ticker'])
    df = df.rename(columns={'Ticker': 'symbol'})
    df['event_timestamp'] = pd.to_datetime(
        df['DateRaw'], format='%Y-%m-%d', errors='coerce', utc=True
    )

    # 2) Predict BUY signals
    model = joblib.load(model_path)
    df['pct_return'] = df.apply(
        lambda r: compute_pct_return(r['symbol'], r['event_timestamp']),
        axis=1
    )
    df['signal'] = model.predict(df[['pct_return']])
    buys = df[df['signal'] == 1].copy()
    if buys.empty:
        return None

    # 3) Fetch entry prices
    buys['entry_price'] = buys.apply(
        lambda r: fetch_price(r['symbol'], r['event_timestamp']), axis=1
    )

    # 4) Save to CSV
    date_str = datetime.utcnow().strftime('%Y-%m-%d')
    folder   = os.path.join(OUTPUT_BASE, name)
    os.makedirs(folder, exist_ok=True)
    out_csv  = os.path.join(folder, f"{name}_signals_{date_str}.csv")
    buys[['symbol','entry_price','event_timestamp']].to_csv(out_csv, index=False)

    # 5) Upload to Drive
    meta  = {'name': os.path.basename(out_csv), 'parents': [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(out_csv, mimetype='text/csv')
    drive_svc.files().create(body=meta, media_body=media).execute()

    return out_csv, buys


# ── TELEGRAM NOTIFICATION ─────────────────────────────────────────────
def send_telegram(file_path, buys, scenario_name):
    now      = datetime.utcnow()
    date_hdr = now.strftime('%d/%m/%y')
    if not buys.empty:
        # Build a single-line bullet per signal, with green‐prefixed numbers
        caption = f"🟢 Scenario: {scenario_name} | BUY Signals {date_hdr} GMT:\n"
        for _, r in buys.iterrows():
            p    = r['entry_price']
            # Calculate entry window
            delta = r['event_timestamp'] - now
            hrs   = max(0, math.ceil(delta.total_seconds() / 3600))
            caption += (
                f"• BUY | Entry Price: 🟢{p if p is not None else 'N/A'} | "
                f"Enter within: 🟢{hrs} hours\n"
            )
            # Schedule your exit reminder (stub)
            schedule_exit_reminder(r['symbol'], r['event_timestamp'], scenario_name)

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        with open(file_path, 'rb') as f:
            requests.post(
                url,
                params={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption},
                files={'document': f}
            )
    else:
        # No signals for this scenario
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={'chat_id': TELEGRAM_CHAT_ID,
                  'text': f"✅ No BUY signals for {scenario_name} today."}
        )


# ── MAIN ───────────────────────────────────────────────────────────────
def main():
    drive = init_drive()
    any_signals = False

    for csv_path in glob.glob("*_history_events.csv"):
        name       = os.path.basename(csv_path).replace("_history_events.csv", "")
        model_path = f"{name}_model.pkl"
        if not os.path.exists(model_path):
            print(f"⚠️  Skipping '{name}': missing model.")
            continue

        res = process_scenario(name, csv_path, model_path, drive)
        if res:
            file_path, buys = res
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
