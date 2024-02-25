# Options Trading Strategy Based on IV

## Project Overview

This project focuses on developing and implementing an options trading strategy grounded in the analysis of implied volatility (IV) term structures. It leverages financial models to recalculate options Greeks, assesses the term structure of IV for different contracts, and executes trades based on these analyses to capitalize on predicted market movements.

### Key Features

- **Greeks Calculation:** Utilizes models like Black-Scholes for real-time calculation of options Greeks.
- **IV Term Structure Analysis:** Analyzes IV across various expiration dates to identify trading opportunities.
- **Strategy Implementation:** Employs strategies based on market conditions (contango or backwardation) to make buy/sell decisions.
- **FIFO Order Tracking:** Manages open positions and accurately calculates P&L using FIFO logic.
- **Realized & Unrealized PnL Calculation:** Evaluates the financial performance of executed trades.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- Pandas
- Any additional libraries used in the project
- The data: https://www.kaggle.com/datasets/kylegraupe/aapl-options-data-2016-2020

### Installation

1. Clone the repository to your local machine:
   ```sh
   git clone https://github.com/zgcharaf/Options-Trading-Framework.git
   ```
2. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

To use this project:
1. Navigate to the project directory.
2. Make sure you download the relevant dataset from Kaggle and place it in the data folder (create if missing)

## Strategy Explanation

The strategy is built on the analysis of IV term structures, distinguishing between contango and backwardation market conditions to predict future volatility. It involves buying or selling options contracts based on these predictions to generate profit.

- **Selling Volatility Forward:** In a contango market, the strategy sells longer-term options and buys shorter-term ones.
- **Buying Volatility Forward:** In a backwardation market, it buys longer-term options and sells shorter-term ones.

## Contributing

Contributions to enhance the project are welcome. Please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Charaf ZGUIOUAR - email@example.com

Project Link: https://github.com/zgcharaf/Options-Trading-Framework

