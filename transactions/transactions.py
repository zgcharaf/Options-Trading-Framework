import pandas as pd 
import numpy as np 

class Transactions:
    def __init__(self):
        self.log_columns = ['DATE', 'DECISION', 'CONTRACT_ID', 'PRICE', 'QUANTITY', 'DTE',
                            'C_DELTA', 'C_GAMMA', 'C_VEGA', 'EXPIRE_DATE', 'UNDERLYING']
        self.transactions_log = pd.DataFrame(columns=self.log_columns)

    def add_transactions(self, transactions_df):
        if not set(transactions_df.columns).issubset(set(self.log_columns)):
            raise ValueError("Input DataFrame columns do not match expected transaction log structure.")
        self.transactions_log = pd.concat([self.transactions_log, transactions_df], ignore_index=True)

    def get_transaction_log(self):
        return self.transactions_log

    def get_transactions_by_date(self, date):
        return self.transactions_log[self.transactions_log['DATE'] == date]

    def get_transactions_by_contract_id(self, contract_id):
        return self.transactions_log[self.transactions_log['CONTRACT_ID'] == contract_id]

    def calculate_net_spent_by_date(self):
     
    # Group by 'DATE' and then apply calculations per group
        grouped = self.transactions_log.groupby('DATE')

    # Define a function to calculate net spent per group
        def net_spent(group):
            buys = group[group['DECISION'] == 'Buy']
            sells = group[group['DECISION'] == 'Sell']
            return (buys['PRICE'] * buys['QUANTITY']).sum() - (sells['PRICE'] * sells['QUANTITY']).sum()

    # Apply the net_spent function to each group and reset index to have 'DATE' as a column
        net_spent_by_date = grouped.apply(net_spent).reset_index(name='NET_SPENT')

        return net_spent_by_date

