"""
StockMarketAI: Core Backtesting Engine
Executes realistic event-driven backtesting with execution delays, slippage,
commissions, stop-loss, take-profit, and benchmark comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.backtesting.metrics import calculate_metrics, format_metrics_table
from src.backtesting.strategy import BaseStrategy, EnhancedHybridStrategy, DirectionStrategy, RegressionExpectedReturnStrategy


class Backtester:
    """
    Realistic Event-Driven Backtesting Simulator.
    
    Execution semantics:
      1. Signal generated at Close of day t using information available up to t.
      2. Orders filled at Open of day t+1 with slippage and commission.
      3. Intra-day Stop Loss and Take Profit monitored via High/Low of day t+1.
      4. End of day mark-to-market portfolio value updated using Close of day t+1.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,   # 0.1% per trade
        slippage_rate: float = 0.0005,    # 0.05% slippage
        results_dir: str = "src/backtesting/results"
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str = "STOCK"
    ) -> dict:
        """
        Run backtest over a chronological DataFrame containing market data and model predictions.

        Required columns in df:
            - Date (as index or column)
            - Open, High, Low, Close, Volume
            - Prediction columns (e.g. direction_prob, pred_Tomorrow_Close, etc.)

        Returns:
            dict containing metrics, trade_log, equity_curve, benchmark_curve.
        """
        data = df.copy()
        if "Date" in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            data["Date"] = pd.to_datetime(data["Date"])
            data.set_index("Date", inplace=True)
        data.sort_index(inplace=True)

        if len(data) < 5:
            raise ValueError("Insufficient data points for backtesting (need at least 5)")

        # Simulation state
        cash = float(self.initial_capital)
        shares = 0
        current_position = None  # Dict with details of active trade

        equity_records = []
        trade_log = []

        dates = data.index
        n_bars = len(data)

        # Pending order from day t to be executed at day t+1 Open
        pending_order = None

        for i in range(n_bars):
            current_date = dates[i]
            row = data.iloc[i]
            prev_row = data.iloc[i-1] if i > 0 else None

            open_p = float(row["Open"])
            high_p = float(row["High"])
            low_p = float(row["Low"])
            close_p = float(row["Close"])

            # ---------------------------------------------------------
            # 1. PROCESS PENDING ORDER FROM PREVIOUS CLOSE (AT TODAY'S OPEN)
            # ---------------------------------------------------------
            if pending_order is not None:
                order_action = pending_order["action"]
                order_reason = pending_order["reason"]
                stop_loss_pct = pending_order.get("stop_loss_pct", strategy.stop_loss_pct)
                take_profit_pct = pending_order.get("take_profit_pct", strategy.take_profit_pct)

                if order_action == "BUY" and shares == 0:
                    # Apply slippage (buy slightly higher)
                    exec_price = open_p * (1.0 + self.slippage_rate)
                    alloc_capital = cash * strategy.position_size_pct
                    raw_shares = int(alloc_capital // (exec_price * (1.0 + self.commission_rate)))

                    if raw_shares > 0:
                        cost = raw_shares * exec_price
                        fee = cost * self.commission_rate
                        cash -= (cost + fee)
                        shares = raw_shares

                        current_position = {
                            "entry_date": current_date,
                            "entry_price": exec_price,
                            "shares": shares,
                            "stop_loss_price": exec_price * (1.0 - stop_loss_pct),
                            "take_profit_price": exec_price * (1.0 + take_profit_pct),
                            "entry_reason": order_reason,
                            "holding_days": 0
                        }

                elif order_action == "SELL" and shares > 0:
                    # Apply slippage (sell slightly lower)
                    exec_price = open_p * (1.0 - self.slippage_rate)
                    gross_revenue = shares * exec_price
                    fee = gross_revenue * self.commission_rate
                    net_revenue = gross_revenue - fee
                    cash += net_revenue

                    pnl = net_revenue - (current_position["shares"] * current_position["entry_price"])
                    pnl_pct = (exec_price - current_position["entry_price"]) / current_position["entry_price"] * 100.0

                    trade_log.append({
                        "symbol": symbol,
                        "entry_date": current_position["entry_date"],
                        "entry_price": round(current_position["entry_price"], 2),
                        "exit_date": current_date,
                        "exit_price": round(exec_price, 2),
                        "shares": shares,
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "holding_days": current_position["holding_days"],
                        "exit_reason": f"Signal Exit ({order_reason})"
                    })

                    shares = 0
                    current_position = None

                pending_order = None

            # ---------------------------------------------------------
            # 2. CHECK INTRA-DAY STOP LOSS & TAKE PROFIT (FOR ACTIVE POSITION)
            # ---------------------------------------------------------
            if shares > 0 and current_position is not None:
                current_position["holding_days"] += 1
                sl_price = current_position["stop_loss_price"]
                tp_price = current_position["take_profit_price"]

                exited_intraday = False
                exit_price = None
                exit_reason = None

                # Check Stop Loss breach
                if low_p <= sl_price:
                    # Gap down handling: if open was below stop loss, fill at open, otherwise stop price
                    exit_price = min(open_p, sl_price) * (1.0 - self.slippage_rate)
                    exit_reason = "Stop Loss Hit"
                    exited_intraday = True
                # Check Take Profit breach
                elif high_p >= tp_price:
                    exit_price = max(open_p, tp_price) * (1.0 - self.slippage_rate)
                    exit_reason = "Take Profit Hit"
                    exited_intraday = True
                # Check Maximum Holding Days
                elif current_position["holding_days"] >= strategy.max_holding_days:
                    exit_price = close_p * (1.0 - self.slippage_rate)
                    exit_reason = f"Max Holding Days ({strategy.max_holding_days}d) Exceeded"
                    exited_intraday = True

                if exited_intraday:
                    gross_revenue = shares * exit_price
                    fee = gross_revenue * self.commission_rate
                    net_revenue = gross_revenue - fee
                    cash += net_revenue

                    pnl = net_revenue - (current_position["shares"] * current_position["entry_price"])
                    pnl_pct = (exit_price - current_position["entry_price"]) / current_position["entry_price"] * 100.0

                    trade_log.append({
                        "symbol": symbol,
                        "entry_date": current_position["entry_date"],
                        "entry_price": round(current_position["entry_price"], 2),
                        "exit_date": current_date,
                        "exit_price": round(exit_price, 2),
                        "shares": shares,
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "holding_days": current_position["holding_days"],
                        "exit_reason": exit_reason
                    })

                    shares = 0
                    current_position = None

            # ---------------------------------------------------------
            # 3. GENERATE SIGNAL AT CLOSE FOR NEXT DAY EXECUTION
            # ---------------------------------------------------------
            if i < n_bars - 1:  # Only if there is a tomorrow to execute
                signal = strategy.generate_signal(row, prev_row)
                if signal["action"] in ["BUY", "SELL"]:
                    pending_order = {
                        "action": signal["action"],
                        "reason": signal["reason"],
                        "stop_loss_pct": signal.get("stop_loss_pct", strategy.stop_loss_pct),
                        "take_profit_pct": signal.get("take_profit_pct", strategy.take_profit_pct)
                    }

            # ---------------------------------------------------------
            # 4. MARK-TO-MARKET PORTFOLIO VALUE AT CLOSE
            # ---------------------------------------------------------
            holdings_val = shares * close_p
            total_equity = cash + holdings_val
            equity_records.append({
                "Date": current_date,
                "Equity": total_equity,
                "Cash": cash,
                "Holdings": holdings_val,
                "Close": close_p
            })

        # Close any open position at the very end of backtest
        if shares > 0 and current_position is not None:
            last_close = float(data.iloc[-1]["Close"])
            gross_revenue = shares * last_close
            fee = gross_revenue * self.commission_rate
            net_revenue = gross_revenue - fee
            cash += net_revenue
            pnl = net_revenue - (current_position["shares"] * current_position["entry_price"])
            pnl_pct = (last_close - current_position["entry_price"]) / current_position["entry_price"] * 100.0

            trade_log.append({
                "symbol": symbol,
                "entry_date": current_position["entry_date"],
                "entry_price": round(current_position["entry_price"], 2),
                "exit_date": dates[-1],
                "exit_price": round(last_close, 2),
                "shares": shares,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "holding_days": current_position["holding_days"],
                "exit_reason": "End of Backtest Period"
            })
            equity_records[-1]["Equity"] = cash
            equity_records[-1]["Holdings"] = 0.0

        # Construct equity series
        equity_df = pd.DataFrame(equity_records).set_index("Date")
        equity_series = equity_df["Equity"]

        # Buy & Hold Benchmark Curve
        first_close = float(data.iloc[0]["Close"])
        benchmark_series = (data["Close"] / first_close) * self.initial_capital

        trades_df = pd.DataFrame(trade_log)

        # Calculate full metrics
        metrics = calculate_metrics(
            equity_curve=equity_series,
            trade_log=trades_df,
            benchmark_equity=benchmark_series
        )

        return {
            "symbol": symbol,
            "strategy_name": strategy.__class__.__name__,
            "metrics": metrics,
            "equity_df": equity_df,
            "equity_series": equity_series,
            "benchmark_series": benchmark_series,
            "trade_log": trades_df
        }

    def plot_results(self, backtest_result: dict, save_filename: str = None) -> str:
        """
        Generate and save a 3-panel backtest performance visualization:
          1. Strategy Equity vs Buy & Hold Benchmark
          2. Drawdown percentage over time
          3. Underlying Stock Price with Buy/Sell execution points
        """
        symbol = backtest_result["symbol"]
        strategy_name = backtest_result["strategy_name"]
        equity_series = backtest_result["equity_series"]
        benchmark_series = backtest_result["benchmark_series"]
        trades_df = backtest_result["trade_log"]
        equity_df = backtest_result["equity_df"]
        metrics = backtest_result["metrics"]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.2, 1.8]})
        plt.subplots_adjust(hspace=0.15)

        # 1. Equity Curve vs Benchmark
        ax1.plot(equity_series.index, equity_series.values, label=f"StockMarketAI Strategy ({strategy_name})", color="#00C853", linewidth=2.0)
        ax1.plot(benchmark_series.index, benchmark_series.values, label=f"Buy & Hold Benchmark ({symbol})", color="#78909C", linestyle="--", linewidth=1.5)
        ax1.set_title(f"StockMarketAI Backtest: {symbol} | Total Return: {metrics['total_return_pct']:+.2f}% vs Benchmark: {metrics['benchmark'].get('benchmark_return_pct', 0):+.2f}% | Sharpe: {metrics['sharpe_ratio']}", fontsize=13, fontweight='bold', pad=10)
        ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", frameon=True)
        ax1.yaxis.set_major_formatter('${x:,.0f}')

        # 2. Drawdowns
        rolling_max = equity_series.cummax()
        strategy_dd = (equity_series - rolling_max) / rolling_max * 100.0
        bench_max = benchmark_series.cummax()
        bench_dd = (benchmark_series - bench_max) / bench_max * 100.0

        ax2.fill_between(strategy_dd.index, strategy_dd.values, 0, color="#E53935", alpha=0.4, label=f"Strategy Drawdown (Max: {metrics['max_drawdown_pct']:.1f}%)")
        ax2.plot(bench_dd.index, bench_dd.values, color="#9E9E9E", linestyle=":", label=f"Benchmark Drawdown (Max: {metrics['benchmark'].get('benchmark_mdd_pct', 0):.1f}%)")
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower left", frameon=True)

        # 3. Stock Price with Trade Markers
        stock_close = equity_df["Close"]
        ax3.plot(stock_close.index, stock_close.values, label=f"{symbol} Close Price", color="#29B6F6", linewidth=1.2)

        if not trades_df.empty:
            buy_dates = pd.to_datetime(trades_df["entry_date"])
            buy_prices = trades_df["entry_price"].values
            sell_dates = pd.to_datetime(trades_df["exit_date"])
            sell_prices = trades_df["exit_price"].values

            ax3.scatter(buy_dates, buy_prices, marker="^", color="#00E676", s=70, label="Buy Entry", zorder=5)
            ax3.scatter(sell_dates, sell_prices, marker="v", color="#FF1744", s=70, label="Sell Exit", zorder=5)

        ax3.set_ylabel("Stock Price ($)", fontsize=11)
        ax3.set_xlabel("Date", fontsize=11)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper left", frameon=True)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        if save_filename is None:
            save_filename = f"backtest_{symbol}_{strategy_name.lower()}.png"

        save_path = os.path.join(self.results_dir, save_filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
