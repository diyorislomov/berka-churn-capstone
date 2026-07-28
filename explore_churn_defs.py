"""
Explore alternative churn definitions for Berka dataset.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src.data_loader import load_transactions, load_accounts

print("Loading data...")
trans    = load_transactions()
accounts = load_accounts()

OBS_START   = pd.Timestamp("1993-01-01")
OBS_END     = pd.Timestamp("1996-12-31")   # shift obs window
LABEL_START = pd.Timestamp("1997-01-01")   # shift label window
LABEL_END   = pd.Timestamp("1997-12-31")

eligible_ids = accounts[accounts['account_open_date'] <= OBS_END]['account_id'].tolist()
print(f"Eligible accounts (opened before {OBS_END.date()}): {len(eligible_ids)}")

obs_trans   = trans[(trans['trans_date'] >= OBS_START) & (trans['trans_date'] <= OBS_END)]
label_trans = trans[(trans['trans_date'] >= LABEL_START) & (trans['trans_date'] <= LABEL_END)]

obs_counts   = obs_trans.groupby('account_id').size().rename('obs_count')
label_counts = label_trans.groupby('account_id').size().rename('label_count')

df = pd.DataFrame({'account_id': eligible_ids})
df = df.merge(obs_counts.reset_index(),   on='account_id', how='left')
df = df.merge(label_counts.reset_index(), on='account_id', how='left')
df['obs_count'].fillna(0, inplace=True)
df['label_count'].fillna(0, inplace=True)

# Historical monthly average (obs window = 48 months)
df['obs_monthly_avg'] = df['obs_count'] / 48.0

# Expected count in labeling window (12 months)
df['expected_count'] = df['obs_monthly_avg'] * 12

# Relative activity in labeling window
df['activity_ratio'] = df.apply(
    lambda r: r['label_count'] / r['expected_count'] if r['expected_count'] > 0 else 0, axis=1
)

print("\n--- Option A: churn = zero transactions in 1997 ---")
churnA = (df['label_count'] == 0).sum()
print(f"  Churned: {churnA} ({churnA/len(df)*100:.1f}%)")

print("\n--- Option B: churn = activity_ratio < 0.30 (fell to <30% of historical) ---")
churnB = (df['activity_ratio'] < 0.30).sum()
print(f"  Churned: {churnB} ({churnB/len(df)*100:.1f}%)")

print("\n--- Option C: churn = activity_ratio < 0.50 (fell to <50% of historical) ---")
churnC = (df['activity_ratio'] < 0.50).sum()
print(f"  Churned: {churnC} ({churnC/len(df)*100:.1f}%)")

print("\n--- Option D: churn = label_count < 6 (fewer than 6 tx in 1997) ---")
churnD = (df['label_count'] < 6).sum()
print(f"  Churned: {churnD} ({churnD/len(df)*100:.1f}%)")

print("\n--- Option E: activity_ratio distribution quantiles ---")
print(df['activity_ratio'].describe(percentiles=[.05,.1,.2,.25,.3,.5]))

print("\n--- label_count distribution ---")
print(df['label_count'].describe(percentiles=[.05,.1,.2,.25]))
print(f"Accounts with 0 tx in 1997: {(df['label_count']==0).sum()}")
print(f"Accounts with <3 tx in 1997: {(df['label_count']<3).sum()}")
print(f"Accounts with <6 tx in 1997: {(df['label_count']<6).sum()}")
print(f"Accounts with <12 tx in 1997: {(df['label_count']<12).sum()}")
