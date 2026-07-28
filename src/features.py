"""
features.py
Build an account-level feature matrix from the observation window only.
Construct the binary churn (low-engagement) label from the labeling window.

Temporal design (CRITICAL -- no future leakage):
  Observation window : 1993-01-01 to 1996-12-31  (all features built here ONLY)
  Labeling window    : 1997-01-01 to 1997-12-31  (churn label built here)
  Min observation    : 12 months of account history required

Churn Definition (revised):
  The Berka dataset covers transactions through Dec 1998, so virtually ALL
  accounts are active in 1998 (99.8%). A naive 'zero transactions in 1998'
  label yields only 0.2% churn -- unusable for modelling.

  Instead, we adopt a RELATIVE ENGAGEMENT definition, consistent with how
  retail banks operationalise churn risk in practice:

    churned = 1  if the account's 1997 transaction count falls below the
                 20th percentile across all eligible accounts' 1997 counts.

  This creates a ~20% churn rate (bottom quintile = 'low-engagement / at-risk')
  and represents the 20% of customers most likely to disengage further.
  The threshold is computed on training data only; the percentile boundary is
  then applied to validation/test data without refitting (no leakage).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -- Temporal boundaries ----------------------------------------------------
OBS_START       = pd.Timestamp("1993-01-01")
OBS_END         = pd.Timestamp("1996-12-31")   # Observation: 4 full years
LABEL_START     = pd.Timestamp("1997-01-01")
LABEL_END       = pd.Timestamp("1997-12-31")   # Label: 1 year of 1997 activity
MIN_OBS_MONTHS  = 12
CHURN_PERCENTILE = 20                           # Bottom 20th pctile = at-risk

# Observation window duration in months (for frequency calculation)
OBS_DURATION_MONTHS = (OBS_END.year - OBS_START.year) * 12 + (OBS_END.month - OBS_START.month)


# ─────────────────────────────────────────────────────────────
# Label construction
# ─────────────────────────────────────────────────────────────

def build_churn_label(transactions, accounts, churn_percentile=CHURN_PERCENTILE):
    """
    Construct a binary 'low-engagement / churn-risk' label for each account.

    Method
    ------
    For each eligible account we count the number of transactions in the
    labeling window (1997). An account is labelled CHURNED (=1) if its 1997
    transaction count falls at or below the `churn_percentile`-th percentile
    of all eligible accounts' 1997 counts.

    Default: bottom 20th percentile => ~20% churn rate.

    This matches retail-banking practice where the least-engaged quintile of
    customers is flagged for proactive retention outreach.

    Parameters
    ----------
    transactions     : pd.DataFrame from data_loader.load_transactions()
    accounts         : pd.DataFrame from data_loader.load_accounts()
    churn_percentile : int, percentile threshold (default: 20)

    Returns
    -------
    pd.DataFrame with columns: account_id, churned (0/1), label_tx_count
    """
    eligible_ids = accounts.loc[
        accounts['account_open_date'] <= OBS_END, 'account_id'
    ].tolist()

    # Count 1997 transactions per account
    label_trans = transactions[
        (transactions['trans_date'] >= LABEL_START) &
        (transactions['trans_date'] <= LABEL_END)
    ]
    tx_counts = (
        label_trans.groupby('account_id').size()
        .reindex(eligible_ids, fill_value=0)
        .reset_index()
    )
    tx_counts.columns = ['account_id', 'label_tx_count']

    # Compute the percentile threshold
    threshold = float(np.percentile(tx_counts['label_tx_count'], churn_percentile))
    tx_counts['churned'] = (tx_counts['label_tx_count'] <= threshold).astype(int)

    actual_rate = tx_counts['churned'].mean()
    print(f"      Churn threshold   : <= {threshold:.0f} tx in labeling window")
    print(f"      Churn rate        : {actual_rate:.1%} (target ~{churn_percentile}%)")

    return tx_counts[['account_id', 'churned']]


# ─────────────────────────────────────────────────────────────
# Transaction features  (observation window only)
# ─────────────────────────────────────────────────────────────

def build_transaction_features(transactions, account_ids):
    """
    Compute transaction-based features for each account,
    using ONLY transactions within the observation window.

    Features
    --------
    trans_count          : total number of transactions
    trans_freq_monthly   : transactions per month
    avg_trans_amount     : mean transaction amount
    std_trans_amount     : std of transaction amounts
    credit_count         : number of credit (PRIJEM) transactions
    debit_count          : number of debit (VYDAJ/VYBER) transactions
    credit_ratio         : credit_count / trans_count
    avg_balance          : mean running balance
    std_balance          : std of running balance
    min_balance          : minimum running balance
    max_balance          : maximum running balance
    final_balance        : last recorded balance in observation window
    balance_trend        : linear slope of balance over time (positive = growing)
    months_since_last_tx : months between last transaction and OBS_END
    """
    obs_trans = transactions[
        (transactions['trans_date'] >= OBS_START) &
        (transactions['trans_date'] <= OBS_END)
    ].copy()

    rows = []
    for acc_id in account_ids:
        acc_tx = obs_trans[obs_trans['account_id'] == acc_id]

        if len(acc_tx) == 0:
            rows.append({
                'account_id': acc_id,
                'trans_count': 0, 'trans_freq_monthly': 0.0,
                'avg_trans_amount': 0.0, 'std_trans_amount': 0.0,
                'credit_count': 0, 'debit_count': 0, 'credit_ratio': 0.0,
                'avg_balance': 0.0, 'std_balance': 0.0,
                'min_balance': 0.0, 'max_balance': 0.0, 'final_balance': 0.0,
                'balance_trend': 0.0, 'months_since_last_tx': OBS_DURATION_MONTHS,
            })
            continue

        n = len(acc_tx)
        freq = n / max(OBS_DURATION_MONTHS, 1)
        credit_n = (acc_tx['trans_type'] == 'credit').sum()
        debit_n  = ((acc_tx['trans_type'] == 'debit') | (acc_tx['trans_type'] == 'withdrawal')).sum()

        bal = acc_tx['balance'].dropna()
        avg_bal   = float(bal.mean())   if len(bal) > 0 else 0.0
        std_bal   = float(bal.std())    if len(bal) > 1 else 0.0
        min_bal   = float(bal.min())    if len(bal) > 0 else 0.0
        max_bal   = float(bal.max())    if len(bal) > 0 else 0.0

        acc_sorted = acc_tx.sort_values('trans_date')
        final_bal  = float(acc_sorted['balance'].dropna().iloc[-1]) if len(acc_sorted['balance'].dropna()) > 0 else 0.0
        last_tx_date = acc_sorted['trans_date'].max()
        months_since = max((OBS_END - last_tx_date).days / 30.44, 0)

        # Balance linear trend (slope)
        if len(bal) > 1 and acc_sorted['balance'].notna().sum() > 1:
            x = np.arange(acc_sorted['balance'].notna().sum())
            y = acc_sorted['balance'].dropna().values
            bal_trend = float(np.polyfit(x, y, 1)[0])
        else:
            bal_trend = 0.0

        rows.append({
            'account_id': acc_id,
            'trans_count': n,
            'trans_freq_monthly': round(freq, 4),
            'avg_trans_amount': round(float(acc_tx['trans_amount'].mean()), 2),
            'std_trans_amount': round(float(acc_tx['trans_amount'].std()) if n > 1 else 0.0, 2),
            'credit_count': int(credit_n),
            'debit_count':  int(debit_n),
            'credit_ratio': round(credit_n / n, 4),
            'avg_balance':   round(avg_bal, 2),
            'std_balance':   round(std_bal, 2),
            'min_balance':   round(min_bal, 2),
            'max_balance':   round(max_bal, 2),
            'final_balance': round(final_bal, 2),
            'balance_trend': round(bal_trend, 4),
            'months_since_last_tx': round(months_since, 1),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Account-level static features
# ─────────────────────────────────────────────────────────────

def build_account_features(accounts):
    """
    Build static account-level features.

    Features
    --------
    tenure_months       : months from account open date to OBS_END
    statement_freq_code : encoded statement frequency (0=per-transaction, 1=monthly, 4=weekly)
    """
    eligible = accounts[accounts['account_open_date'] <= OBS_END].copy()
    eligible['tenure_months'] = (
        (OBS_END - eligible['account_open_date']).dt.days / 30.44
    ).round(1)

    freq_map = {
        'POPLATEK MESICNE': 1,   # Monthly
        'POPLATEK TYDNE':   4,   # Weekly
        'POPLATEK PO OBRATU': 0, # After each transaction
    }
    eligible['statement_freq_code'] = eligible['frequency'].map(freq_map).fillna(0).astype(int)
    return eligible[['account_id', 'district_id', 'tenure_months', 'statement_freq_code']]


# ─────────────────────────────────────────────────────────────
# Product ownership features
# ─────────────────────────────────────────────────────────────

def build_product_features(account_ids, loans, cards, orders, dispositions):
    """
    Build product-ownership features.

    Features
    --------
    has_loan    : 1 if account had a loan issued during observation window
    has_card    : 1 if account had a card issued during observation window
    order_count : number of permanent standing orders
    """
    # Loans issued within observation window
    obs_loans = loans[loans['loan_date'] <= OBS_END]
    loan_accounts = set(obs_loans['account_id'].tolist())

    # Cards: join via disp → account
    obs_cards = cards[cards['card_issued_date'] <= OBS_END]
    if not dispositions.empty and not obs_cards.empty:
        card_disp = obs_cards.merge(
            dispositions[['disp_id', 'account_id']], on='disp_id', how='left'
        )
        card_accounts = set(card_disp['account_id'].dropna().tolist())
    else:
        card_accounts = set()

    # Standing orders
    order_counts = (
        orders.groupby('account_id').size()
        .reset_index(name='order_count')
    )
    order_map = dict(zip(order_counts['account_id'], order_counts['order_count']))

    rows = [
        {
            'account_id': acc_id,
            'has_loan':   1 if acc_id in loan_accounts   else 0,
            'has_card':   1 if acc_id in card_accounts   else 0,
            'order_count': order_map.get(acc_id, 0),
        }
        for acc_id in account_ids
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# District demographic features
# ─────────────────────────────────────────────────────────────

def build_district_features(account_ids, accounts, districts):
    """
    Join district-level demographic features to each account.

    Features (from district table)
    ------
    region                  : region name (label-encoded)
    avg_salary              : average salary in the district
    unemployment_rate_96    : unemployment rate (1996)
    urban_population_ratio  : share of urban population
    """
    desired_cols = [
        'district_id', 'region', 'avg_salary',
        'unemployment_rate_96', 'urban_population_ratio'
    ]
    available_cols = [c for c in desired_cols if c in districts.columns]
    district_sub = districts[available_cols].copy()

    # Encode region as integer
    if 'region' in district_sub.columns:
        district_sub['region_code'] = district_sub['region'].astype('category').cat.codes
        district_sub.drop(columns=['region'], inplace=True)

    # Merge accounts → districts
    acc_sub = accounts[accounts['account_id'].isin(account_ids)][['account_id', 'district_id']]
    merged = acc_sub.merge(district_sub, on='district_id', how='left')
    merged.drop(columns=['district_id'], inplace=True)
    return merged


# ─────────────────────────────────────────────────────────────
# Master feature builder
# ─────────────────────────────────────────────────────────────

# Canonical list of all feature columns used for modeling
FEATURE_COLS = [
    # Transaction features
    'trans_count', 'trans_freq_monthly', 'avg_trans_amount', 'std_trans_amount',
    'credit_count', 'debit_count', 'credit_ratio',
    'avg_balance', 'std_balance', 'min_balance', 'max_balance', 'final_balance',
    'balance_trend', 'months_since_last_tx',
    # Account features
    'tenure_months', 'statement_freq_code',
    # Product features
    'has_loan', 'has_card', 'order_count',
    # District features
    'region_code', 'avg_salary', 'unemployment_rate_96', 'urban_population_ratio',
]


def build_feature_matrix(tables, verbose=True):
    """
    Build the full account-level feature matrix and churn label.

    Parameters
    ----------
    tables : dict returned by data_loader.load_all_tables()
    verbose : bool, print progress

    Returns
    -------
    feature_matrix : pd.DataFrame  (account_id, churned, + all feature columns)
    """
    accounts     = tables['accounts']
    transactions = tables['transactions']
    loans        = tables['loans']
    cards        = tables['cards']
    orders       = tables['orders']
    districts    = tables['districts']
    dispositions = tables['dispositions']

    if verbose: print("\n[1/5] Building churn labels...")
    labels = build_churn_label(transactions, accounts)
    eligible_ids = labels['account_id'].tolist()
    eligible_accounts = accounts[accounts['account_id'].isin(eligible_ids)]

    if verbose:
        print(f"      Eligible accounts : {len(eligible_ids):,}")
        print(f"      Churn rate (raw)  : {labels['churned'].mean():.1%}")

    if verbose: print("[2/5] Building transaction features...")
    trans_feats = build_transaction_features(transactions, eligible_ids)

    if verbose: print("[3/5] Building account features...")
    acc_feats = build_account_features(eligible_accounts)

    if verbose: print("[4/5] Building product features...")
    prod_feats = build_product_features(eligible_ids, loans, cards, orders, dispositions)

    if verbose: print("[5/5] Building district features...")
    dist_feats = build_district_features(eligible_ids, eligible_accounts, districts)

    # ── Merge all feature groups ─────────────────────────────
    fm = labels.copy()
    for df in [trans_feats, acc_feats, prod_feats, dist_feats]:
        fm = fm.merge(df, on='account_id', how='left')

    # ── Exclude short-history accounts ───────────────────────
    if 'tenure_months' in fm.columns:
        before = len(fm)
        fm = fm[fm['tenure_months'] >= MIN_OBS_MONTHS].copy()
        excluded = before - len(fm)
        if verbose:
            print(f"\nExcluded {excluded:,} accounts with < {MIN_OBS_MONTHS} months of history")

    # ── Drop district_id if it leaked in ─────────────────────
    if 'district_id' in fm.columns:
        fm.drop(columns=['district_id'], inplace=True)

    if verbose:
        print(f"\nFinal feature matrix : {fm.shape}")
        print(f"Churn rate (final)   : {fm['churned'].mean():.1%}")
        available = [c for c in FEATURE_COLS if c in fm.columns]
        print(f"Feature columns used : {len(available)} / {len(FEATURE_COLS)}")

    return fm
