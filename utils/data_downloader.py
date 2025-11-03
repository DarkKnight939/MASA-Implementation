"""
Download market data from Yahoo Finance and prepare CSVs matching the project's sample format.

Functions
- download_market_data(config, tickers=None)

Assumptions
- If `tickers` is None, a default list for `DJIA` (10 tickers) will be used. You can pass a custom list.
"""

import os
import datetime
import pandas as pd


def _default_tickers_for_market(market_name, topK):
    """Return reasonable default 10-ticker subsets for common indices."""
    if market_name == 'DJIA':
        lst = ['AAPL', 'MSFT', 'JPM', 'V', 'INTC', 'CSCO', 'WMT', 'UNH', 'HD', 'CAT']
    elif market_name == 'SP500':
        lst = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']
    else:
        lst = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']
    return lst[:topK]


def _safe_int(x):
    """Safely convert volume to int; NaNs become 0."""
    try:
        if pd.isna(x):
            return 0
        return int(x)
    except Exception:
        return 0


def download_market_data(config, tickers=None, save_dir=None):
    """Download OHLCV data for `tickers` between the config train/test date range."""
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError(
            "yfinance is required to download market data. Please install it using:\n  pip install yfinance"
        ) from e

    # Ensure save directory
    if save_dir is None:
        save_dir = config.dataDir
    os.makedirs(save_dir, exist_ok=True)

    # Choose tickers
    tickers = tickers or _default_tickers_for_market(config.market_name, config.topK)
    tickers = tickers[:config.topK]

    # Determine date range
    start = getattr(config, "train_date_start", datetime.datetime(2010, 1, 1))
    end = getattr(config, "test_date_end", datetime.datetime.now())

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print(f"Downloading {len(tickers)} tickers from {start_str} to {end_str}...")

    # Main stock data
    data = yf.download(
        tickers,
        start=start_str,
        end=end_str,
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )

    rows, index_rows = [], []

    for idx, tic in enumerate(tickers, start=1):
        try:
            df = data[tic].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
        except Exception:
            df = data.copy()

        if df is None or df.empty:
            print(f"⚠️ Warning: No data for {tic}")
            continue

        # Keep only OHLCV
        if not set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns):
            print(f"⚠️ Warning: {tic} missing OHLCV columns, skipping.")
            continue

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"])
        df["Volume"] = df["Volume"].apply(_safe_int)

        for dtt, row in df.iterrows():
            rows.append(
                {
                    "date": dtt.strftime("%m/%d/%Y"),
                    "stock": idx,
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"],
                }
            )
        index_rows.append({"stock": idx, "tic": tic})

    if not rows:
        raise RuntimeError("❌ No market data was downloaded for the requested tickers/date range.")

    out_df = pd.DataFrame(rows)
    out_df["__dt"] = pd.to_datetime(out_df["date"])
    out_df = out_df.sort_values(["__dt", "stock"]).drop(columns="__dt")

    fname = f"{config.market_name}_{config.topK}_{config.freq}.csv"
    fpath = os.path.join(save_dir, fname)
    out_df.to_csv(fpath, index=False)
    print(f"✅ Saved combined data to {fpath}")

    # Market index
    market_index_ticker = {
        "DJIA": "^DJI",
        "SP500": "^GSPC",
        "CSI300": "000300.SS",
    }.get(config.market_name)

    if market_index_ticker:
        idx_data = yf.download(
            market_index_ticker,
            start=start_str,
            end=end_str,
            progress=False,
            auto_adjust=False,
        )
        if not idx_data.empty:
            idx_data = idx_data[["Open", "High", "Low", "Close", "Volume"]].copy()
            idx_data["Volume"] = idx_data["Volume"].apply(_safe_int)

            idx_data["date"] = idx_data.index.strftime("%m/%d/%Y")
            idx_df = idx_data.reset_index(drop=True)

            idx_fname = f"{config.market_name}_{config.freq}_index.csv"
            idx_fpath = os.path.join(save_dir, idx_fname)
            idx_df.to_csv(idx_fpath, index=False)
            print(f"✅ Saved market index data to {idx_fpath}")
    else:
        print(f"ℹ️ No market index ticker defined for {config.market_name}, skipping index download.")

    # Stock–ticker mapping
    map_fname = f"{config.market_name}_{config.topK}_{config.freq}_mapping.csv"
    map_fpath = os.path.join(save_dir, map_fname)
    pd.DataFrame(index_rows).to_csv(map_fpath, index=False)
    print(f"✅ Saved stock/ticker mapping to {map_fpath}")

    return fpath
