import sys
sys.path.insert(0, '.')
import pandas as pd
from src.data_loader import load_transactions, load_accounts
from src.features import OBS_END, LABEL_START, LABEL_END

print('Loading transactions...')
trans = load_transactions()
accounts = load_accounts()

print('Transaction date range:', trans['trans_date'].min(), 'to', trans['trans_date'].max())
print('Total transactions:', len(trans))

label_trans = trans[(trans['trans_date'] >= LABEL_START) & (trans['trans_date'] <= LABEL_END)]
print(f'Transactions in labeling window 1998: {len(label_trans):,}')

active_in_1998 = label_trans['account_id'].nunique()
total_accounts  = accounts['account_id'].nunique()
print(f'Accounts active in 1998: {active_in_1998:,} out of {total_accounts:,}')

last_tx = trans.groupby('account_id')['trans_date'].max().reset_index()
last_tx.columns = ['account_id', 'last_tx_date']
print()
print('Last transaction date distribution:')
print(last_tx['last_tx_date'].describe())
print()

before_1998 = last_tx[last_tx['last_tx_date'] < LABEL_START]
print(f'Accounts with last tx BEFORE 1998: {len(before_1998)} ({len(before_1998)/len(last_tx)*100:.1f}%)')

last_tx['last_tx_year'] = last_tx['last_tx_date'].dt.year
print()
print('Last transaction year distribution:')
print(last_tx['last_tx_year'].value_counts().sort_index())

# Check what a more lenient churn definition looks like
print()
print('--- Churn rate sensitivity ---')
for cutoff_month in [3, 6, 9]:
    cutoff = pd.Timestamp(f'1997-{cutoff_month:02d}-01')
    inactive = last_tx[last_tx['last_tx_date'] < cutoff]
    print(f'Accounts with last tx before {cutoff.date()}: {len(inactive)} ({len(inactive)/len(last_tx)*100:.1f}%)')
