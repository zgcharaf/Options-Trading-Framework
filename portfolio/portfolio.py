import pandas as pd 
import numpy as np 

class Portfolio:
    def __init__(self, options_prices_df, transactions_log):
        self.options_prices_df = options_prices_df
        self.transactions_log = transactions_log

    def calculate_fifo_pnl(self):
        open_positions = []
        pnl_records = []

        for quote_date in sorted(self.options_prices_df['QUOTE_DATE'].unique()):
            transactions_up_to_date = self.transactions_log[self.transactions_log['DATE'] <= quote_date]
            market_data_on_date = self.options_prices_df[self.options_prices_df['QUOTE_DATE'] == quote_date]

            daily_realized_pnl = 0

            for _, transaction in transactions_up_to_date.iterrows():
                contract_id = transaction['CONTRACT_ID']
                decision = transaction['DECISION']
                quantity = transaction['QUANTITY']
                price = transaction['PRICE']

                if decision == 'Buy':
                    # Check for existing SHORT positions to cover
                    for op in list(open_positions):
                        if op['CONTRACT_ID'] == contract_id and op['TYPE'] == 'SHORT' and quantity > 0:
                            cover_quantity = min(quantity, op['QUANTITY'])
                            realized_pnl = (op['PRICE'] - price) * cover_quantity  # Profit from covering shorts
                            daily_realized_pnl += realized_pnl
                            quantity -= cover_quantity
                            op['QUANTITY'] -= cover_quantity
                            if op['QUANTITY'] == 0:
                                open_positions.remove(op)
                    if quantity > 0:  # Any remaining quantity is a new LONG position
                        open_positions.append({'CONTRACT_ID': contract_id, 'QUANTITY': quantity, 'PRICE': price, 'TYPE': 'LONG'})
                else:  # Sell transaction
                    # Check for existing LONG positions to cover
                    for op in list(open_positions):
                        if op['CONTRACT_ID'] == contract_id and op['TYPE'] == 'LONG' and quantity > 0:
                            cover_quantity = min(quantity, op['QUANTITY'])
                            realized_pnl = (price - op['PRICE']) * cover_quantity
                            daily_realized_pnl += realized_pnl
                            quantity -= cover_quantity
                            op['QUANTITY'] -= cover_quantity
                            if op['QUANTITY'] == 0:
                                open_positions.remove(op)
                    if quantity > 0:  # Any remaining quantity opens a new SHORT position
                        open_positions.append({'CONTRACT_ID': contract_id, 'QUANTITY': quantity, 'PRICE': price, 'TYPE': 'SHORT'})

            # Calculate unrealized P&L for remaining open positions
            daily_unrealized_pnl = 0
            for position in open_positions:
                current_market_data = market_data_on_date[market_data_on_date['Option_Contract_ID'] == position['CONTRACT_ID']]
                if not current_market_data.empty and position['QUANTITY'] > 0:
                    if position['TYPE'] == 'LONG':
                        current_bid_price = current_market_data['C_BID'].iloc[0]
                        unrealized_pnl = (current_bid_price - position['PRICE']) * position['QUANTITY']
                    else:  # SHORT position valuation using ask price
                        current_ask_price = current_market_data['C_ASK'].iloc[0]
                        unrealized_pnl = (position['PRICE'] - current_ask_price) * position['QUANTITY']
                    daily_unrealized_pnl += unrealized_pnl

            pnl_records.append({'QUOTE_DATE': quote_date, 'REALIZED_PNL': daily_realized_pnl, 'UNREALIZED_PNL': daily_unrealized_pnl})

        # Clean up positions with zero quantity
        open_positions = [op for op in open_positions if op['QUANTITY'] > 0]

        return pd.DataFrame(pnl_records)
