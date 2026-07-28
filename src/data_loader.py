"""
data_loader.py
Load and join all 8 Berka dataset tables into account-level DataFrames.
Handles Czech YYMMDD date parsing and all table join logic.

Berka Dataset tables (semicolon-delimited .asc files):
  account.asc   - account_id, district_id, frequency, date
  trans.asc     - trans_id, account_id, date, type, operation, amount, balance, k_symbol, bank, account
  loan.asc      - loan_id, account_id, date, amount, duration, payments, status
  card.asc      - card_id, disp_id, type, issued
  client.asc    - client_id, birth_number, district_id
  disp.asc      - disp_id, client_id, account_id, type
  order.asc     - order_id, account_id, bank_to, account_to, amount, k_symbol
  district.asc  - A1..A16 columns (district demographics)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def _resolve_file(data_dir: Path, stem: str) -> Path:
    """
    Resolve the dataset file path, trying .csv first then .asc.
    This makes the loader work with both the Kaggle CSV version
    and the original PKDD .asc version of the Berka dataset.
    """
    for ext in [".csv", ".asc"]:
        candidate = data_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find '{stem}.csv' or '{stem}.asc' in {data_dir}. "
        f"Please place the Berka dataset files in data/raw/ — see data/README.md."
    )


# ─────────────────────────────────────────────────────────────
# Date parsing utilities
# ─────────────────────────────────────────────────────────────

def parse_czech_date(val):
    """
    Parse Czech YYMMDD integer/string to pd.Timestamp.
    E.g., 960101 -> 1996-01-01.
    """
    try:
        s = str(int(float(val))).zfill(6)
        yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
        year = 1900 + yy
        return pd.Timestamp(year=year, month=mm, day=dd)
    except Exception:
        return pd.NaT


def parse_birth_number(bn):
    """
    Parse Czech birth_number to (date_of_birth, gender).
    Format: YYMMDD for male, YY(MM+50)DD for female.
    Returns: (pd.Timestamp or NaT, 'M'/'F')
    """
    try:
        s = str(int(float(bn))).zfill(6)
        yy, mm_raw, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
        if mm_raw > 50:
            gender = 'F'
            mm = mm_raw - 50
        else:
            gender = 'M'
            mm = mm_raw
        year = 1900 + yy
        dob = pd.Timestamp(year=year, month=mm, day=dd)
        return dob, gender
    except Exception:
        return pd.NaT, 'Unknown'


# ─────────────────────────────────────────────────────────────
# Individual table loaders
# ─────────────────────────────────────────────────────────────

def load_accounts(data_dir=None):
    """Load account file (.csv or .asc). Returns DataFrame with parsed open date."""
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    df = pd.read_csv(_resolve_file(data_dir, "account"), sep=";", low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    df['date'] = df['date'].apply(parse_czech_date)
    df.rename(columns={'date': 'account_open_date'}, inplace=True)
    return df


def load_transactions(data_dir=None):
    """
    Load trans file (.csv or .asc). Returns DataFrame with:
    - parsed trans_date
    - trans_type mapped from Czech codes to English
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    df = pd.read_csv(_resolve_file(data_dir, "trans"), sep=";", low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    df['date'] = df['date'].apply(parse_czech_date)
    df.rename(columns={
        'date':   'trans_date',
        'type':   'trans_type',
        'amount': 'trans_amount'
    }, inplace=True)
    # Map Czech type codes to English
    type_map = {
        'PRIJEM': 'credit',
        'VYDAJ':  'debit',
        'VYBER':  'withdrawal'
    }
    df['trans_type'] = df['trans_type'].map(type_map).fillna('other')
    # Ensure numeric
    df['trans_amount'] = pd.to_numeric(df['trans_amount'], errors='coerce').fillna(0)
    df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
    return df


def load_loans(data_dir=None):
    """Load loan file (.csv or .asc). Returns DataFrame with parsed loan date."""
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    df = pd.read_csv(_resolve_file(data_dir, "loan"), sep=";", low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    df['date'] = df['date'].apply(parse_czech_date)
    df.rename(columns={'date': 'loan_date'}, inplace=True)
    return df


def load_cards(data_dir=None):
    """
    Load card file (.csv or .asc). Returns DataFrame with parsed card issued date.
    Note: 'issued' field is YYMMDD — take first 6 chars to be safe.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    df = pd.read_csv(_resolve_file(data_dir, "card"), sep=";", low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    df['issued'] = df['issued'].astype(str).str.strip().str[:6].apply(
        lambda x: parse_czech_date(x) if x.isdigit() else pd.NaT
    )
    df.rename(columns={'issued': 'card_issued_date', 'type': 'card_type'}, inplace=True)
    return df


def load_clients(data_dir=None):
    """Load client file (.csv or .asc). Parses birth_number into dob and gender columns."""
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    df = pd.read_csv(_resolve_file(data_dir, "client"), sep=";", low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    parsed = df['birth_number'].apply(parse_birth_number)
    df['dob']    = [p[0] for p in parsed]
    df['gender'] = [p[1] for p in parsed]
    return df


def load_dispositions(data_dir=None):
    """Load disp file (.csv or .asc) — maps clients to accounts with role (OWNER / DISPONENT)."""
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    df = pd.read_csv(_resolve_file(data_dir, "disp"), sep=";", low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    df.rename(columns={'type': 'disp_type'}, inplace=True)
    return df


def load_orders(data_dir=None):
    """Load order file (.csv or .asc) — permanent standing orders (recurring payments)."""
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    df = pd.read_csv(_resolve_file(data_dir, "order"), sep=";", low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    return df


def load_districts(data_dir=None):
    """
    Load district file (.csv or .asc). Renames cryptic A1..A16 columns to meaningful names.
    Handles '?' missing value encoding. The CSV version already has English column names.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    df = pd.read_csv(_resolve_file(data_dir, "district"), sep=";", low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    col_map = {
        'a1':  'district_id',
        'a2':  'district_name',
        'a3':  'region',
        'a4':  'population',
        'a5':  'num_muni_lt499',
        'a6':  'num_muni_500_1999',
        'a7':  'num_muni_2000_9999',
        'a8':  'num_muni_gt10000',
        'a9':  'num_cities',
        'a10': 'urban_population_ratio',
        'a11': 'avg_salary',
        'a12': 'unemployment_rate_95',
        'a13': 'unemployment_rate_96',
        'a14': 'entrepreneurs_per_1000',
        'a15': 'num_crimes_95',
        'a16': 'num_crimes_96',
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Handle '?' missing values in numeric columns
    numeric_cols = [
        'population', 'urban_population_ratio', 'avg_salary',
        'unemployment_rate_95', 'unemployment_rate_96',
        'entrepreneurs_per_1000', 'num_crimes_95', 'num_crimes_96'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('?', '', regex=False), errors='coerce')

    return df


# ─────────────────────────────────────────────────────────────
# Master loader
# ─────────────────────────────────────────────────────────────

def load_all_tables(data_dir=None):
    """
    Load all 8 Berka tables and return as a dictionary.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing the .asc files. Defaults to data/raw/.

    Returns
    -------
    dict with keys: 'accounts', 'transactions', 'loans', 'cards',
                    'clients', 'dispositions', 'orders', 'districts'
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)

    print(f"Loading Berka dataset from: {data_dir}")
    loaders = {
        'accounts':     load_accounts,
        'transactions': load_transactions,
        'loans':        load_loans,
        'cards':        load_cards,
        'clients':      load_clients,
        'dispositions': load_dispositions,
        'orders':       load_orders,
        'districts':    load_districts,
    }

    tables = {}
    for name, loader_fn in loaders.items():
        try:
            tables[name] = loader_fn(data_dir)
            print(f"  [OK]  {name:<15}: {tables[name].shape}")
        except FileNotFoundError as e:
            print(f"  [ERR] {name:<15}: FILE NOT FOUND -- {e}")
            tables[name] = pd.DataFrame()

    return tables


def get_table_summary(tables):
    """Print a summary of all loaded tables."""
    print("\n" + "="*55)
    print(f"{'Table':<18} {'Rows':>8} {'Cols':>6} {'Missing%':>10}")
    print("="*55)
    for name, df in tables.items():
        if df.empty:
            print(f"  {name:<16} {'EMPTY':>8}")
            continue
        missing_pct = df.isnull().mean().mean() * 100
        print(f"  {name:<16} {len(df):>8,} {len(df.columns):>6}  {missing_pct:>8.1f}%")
    print("="*55)
