"""
StockMarketAI: Performance Metrics Calculation Module
Comprehensive quantitative finance metrics for backtesting trading strategies.
"""

import numpy as np
import pandas as pd


def calculate_metrics(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame,
    benchmark_equity: pd.Series = None,
    risk_free_rate: float = 0.04,
    trading_days_per_year: int = 252
) -> dict:
    """
    Calculate comprehensive backtesting performance metrics.

    Args:
        equity_curve: pd.Series of daily portfolio equity values indexed by date.
        trade_log: pd.DataFrame containing trade execution records.
        benchmark_equity: pd.Series of buy-and-hold benchmark equity values.
        risk_free_rate: Annualized risk-free rate (e.g. 0.04 = 4%).
        trading_days_per_year: Typically 252 for US stock markets.

    Returns:
        dict: Performance metrics summary.
    """
    if len(equity_curve) < 2:
        return {"error": "Insufficient equity data for metrics calculation"}

    initial_capital = float(equity_curve.iloc[0])
    final_equity = float(equity_curve.iloc[-1])
    total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0

    # Number of days & years
    num_days = len(equity_curve)
    num_years = max(num_days / trading_days_per_year, 1.0 / trading_days_per_year)

    # CAGR (Compound Annual Growth Rate)
    if final_equity > 0:
        cagr_pct = (((final_equity / initial_capital) ** (1.0 / num_years)) - 1.0) * 100.0
    else:
        cagr_pct = -100.0

    # Daily returns
    daily_returns = equity_curve.pct_change().dropna()
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    annualized_volatility_pct = (std_daily_return * np.sqrt(trading_days_per_year)) * 100.0 if std_daily_return > 0 else 0.0

    # Sharpe Ratio
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / trading_days_per_year) - 1.0
    excess_daily_returns = daily_returns - daily_rf
    if std_daily_return > 0:
        sharpe_ratio = float((excess_daily_returns.mean() / std_daily_return) * np.sqrt(trading_days_per_year))
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio (downside deviation only)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    if downside_std > 0:
        sortino_ratio = float((excess_daily_returns.mean() / downside_std) * np.sqrt(trading_days_per_year))
    else:
        sortino_ratio = float('inf') if mean_daily_return > 0 else 0.0

    # Maximum Drawdown (MDD)
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown_pct = float(drawdowns.min() * 100.0)

    # Longest Drawdown Duration (in days)
    is_in_drawdown = drawdowns < 0
    drawdown_streak = 0
    max_drawdown_duration = 0
    for in_dd in is_in_drawdown:
        if in_dd:
            drawdown_streak += 1
            if drawdown_streak > max_drawdown_duration:
                max_drawdown_duration = drawdown_streak
        else:
            drawdown_streak = 0

    # Calmar Ratio
    calmar_ratio = float(cagr_pct / abs(max_drawdown_pct)) if abs(max_drawdown_pct) > 1e-4 else 0.0

    # Benchmark comparison
    benchmark_metrics = {}
    if benchmark_equity is not None and len(benchmark_equity) >= 2:
        bench_initial = float(benchmark_equity.iloc[0])
        bench_final = float(benchmark_equity.iloc[-1])
        bench_return_pct = ((bench_final - bench_initial) / bench_initial) * 100.0
        bench_cagr_pct = (((bench_final / bench_initial) ** (1.0 / num_years)) - 1.0) * 100.0 if bench_final > 0 else -100.0
        
        bench_daily = benchmark_equity.pct_change().dropna()
        bench_rolling_max = benchmark_equity.cummax()
        bench_drawdowns = (benchmark_equity - bench_rolling_max) / bench_rolling_max
        bench_mdd_pct = float(bench_drawdowns.min() * 100.0)
        
        # Beta & Alpha calculation
        covariance = np.cov(daily_returns, bench_daily)[0][1] if len(daily_returns) == len(bench_daily) else 0.0
        bench_variance = bench_daily.var()
        beta = float(covariance / bench_variance) if bench_variance > 0 else 1.0
        alpha_annualized_pct = float(cagr_pct - (risk_free_rate * 100 + beta * (bench_cagr_pct - risk_free_rate * 100)))

        benchmark_metrics = {
            "benchmark_return_pct": round(bench_return_pct, 2),
            "benchmark_cagr_pct": round(bench_cagr_pct, 2),
            "benchmark_mdd_pct": round(bench_mdd_pct, 2),
            "alpha_annualized_pct": round(alpha_annualized_pct, 2),
            "beta": round(beta, 3),
            "outperformance_pct": round(total_return_pct - bench_return_pct, 2)
        }

    # Trade statistics
    trade_metrics = {}
    if trade_log is not None and not trade_log.empty and "pnl" in trade_log.columns:
        total_trades = len(trade_log)
        winning_trades = trade_log[trade_log["pnl"] > 0]
        losing_trades = trade_log[trade_log["pnl"] < 0]
        even_trades = trade_log[trade_log["pnl"] == 0]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        win_rate_pct = (num_wins / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = float(winning_trades["pnl"].sum()) if num_wins > 0 else 0.0
        gross_loss = abs(float(losing_trades["pnl"].sum())) if num_losses > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

        avg_trade_pnl = float(trade_log["pnl"].mean()) if total_trades > 0 else 0.0
        avg_trade_pct = float(trade_log["pnl_pct"].mean()) if "pnl_pct" in trade_log.columns else 0.0
        avg_win = float(winning_trades["pnl"].mean()) if num_wins > 0 else 0.0
        avg_loss = float(losing_trades["pnl"].mean()) if num_losses > 0 else 0.0
        win_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 0 else float('inf')

        # Consecutive wins / losses
        pnl_signs = (trade_log["pnl"] > 0).astype(int).tolist()
        max_cons_wins = 0
        curr_cons_wins = 0
        max_cons_losses = 0
        curr_cons_losses = 0

        for sign in pnl_signs:
            if sign == 1:
                curr_cons_wins += 1
                curr_cons_losses = 0
                max_cons_wins = max(max_cons_wins, curr_cons_wins)
            else:
                curr_cons_losses += 1
                curr_cons_wins = 0
                max_cons_losses = max(max_cons_losses, curr_cons_losses)

        # Average holding duration (if holding_days present)
        avg_holding_days = float(trade_log["holding_days"].mean()) if "holding_days" in trade_log.columns else 0.0

        trade_metrics = {
            "total_trades": total_trades,
            "winning_trades": num_wins,
            "losing_trades": num_losses,
            "even_trades": len(even_trades),
            "win_rate_pct": round(win_rate_pct, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf",
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "avg_trade_pnl": round(avg_trade_pnl, 2),
            "avg_trade_pct": round(avg_trade_pct, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "win_loss_ratio": round(win_loss_ratio, 2) if win_loss_ratio != float('inf') else "inf",
            "max_consecutive_wins": max_cons_wins,
            "max_consecutive_losses": max_cons_losses,
            "avg_holding_days": round(avg_holding_days, 1)
        }
    else:
        trade_metrics = {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0
        }

    return {
        "initial_capital": round(initial_capital, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "annualized_volatility_pct": round(annualized_volatility_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2) if sortino_ratio != float('inf') else "inf",
        "calmar_ratio": round(calmar_ratio, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "max_drawdown_duration_days": int(max_drawdown_duration),
        "trading_days": num_days,
        "benchmark": benchmark_metrics,
        "trades": trade_metrics
    }


def format_metrics_table(metrics: dict, title: str = "Backtest Performance Summary") -> str:
    """Format metrics dictionary into a clean Markdown table."""
    lines = [
        f"### {title}",
        "",
        "| Metric | Strategy Value | Benchmark (Buy & Hold) |",
        "| :--- | :--- | :--- |",
        f"| **Initial Capital** | ${metrics['initial_capital']:,.2f} | ${metrics['initial_capital']:,.2f} |",
        f"| **Final Equity** | ${metrics['final_equity']:,.2f} | " + (f"${metrics['initial_capital'] * (1 + metrics['benchmark'].get('benchmark_return_pct', 0)/100):,.2f}" if 'benchmark' in metrics and metrics['benchmark'] else "N/A") + " |",
        f"| **Total Return** | **{metrics['total_return_pct']:+.2f}%** | " + (f"**{metrics['benchmark'].get('benchmark_return_pct', 0):+.2f}%**" if 'benchmark' in metrics and metrics['benchmark'] else "N/A") + " |",
        f"| **Annualized Return (CAGR)** | {metrics['cagr_pct']:+.2f}% | " + (f"{metrics['benchmark'].get('benchmark_cagr_pct', 0):+.2f}%" if 'benchmark' in metrics and metrics['benchmark'] else "N/A") + " |",
        f"| **Max Drawdown (MDD)** | {metrics['max_drawdown_pct']:.2f}% | " + (f"{metrics['benchmark'].get('benchmark_mdd_pct', 0):.2f}%" if 'benchmark' in metrics and metrics['benchmark'] else "N/A") + " |",
        f"| **Sharpe Ratio (Rf=4%)** | {metrics['sharpe_ratio']} | N/A |",
        f"| **Sortino Ratio** | {metrics['sortino_ratio']} | N/A |",
        f"| **Calmar Ratio** | {metrics['calmar_ratio']} | N/A |",
        f"| **Annualized Volatility** | {metrics['annualized_volatility_pct']:.2f}% | N/A |",
    ]

    bench = metrics.get('benchmark', {})
    if bench:
        lines.append(f"| **Alpha (Annualized)** | {bench.get('alpha_annualized_pct', 0):+.2f}% | 0.00% |")
        lines.append(f"| **Beta** | {bench.get('beta', 1.0)} | 1.000 |")
        lines.append(f"| **Net Outperformance** | **{bench.get('outperformance_pct', 0):+.2f}%** | 0.00% |")

    trades = metrics.get('trades', {})
    if trades and trades.get('total_trades', 0) > 0:
        lines.extend([
            "",
            "#### Trade Execution Statistics",
            "",
            "| Trade Metric | Value |",
            "| :--- | :--- |",
            f"| **Total Trades** | {trades['total_trades']} |",
            f"| **Winning / Losing Trades** | {trades['winning_trades']} wins / {trades['losing_trades']} losses |",
            f"| **Win Rate** | **{trades['win_rate_pct']:.1f}%** |",
            f"| **Profit Factor** | {trades['profit_factor']} |",
            f"| **Gross Profit / Loss** | +${trades['gross_profit']:,.2f} / -${trades['gross_loss']:,.2f} |",
            f"| **Average Trade P&L** | ${trades['avg_trade_pnl']:+,.2f} ({trades['avg_trade_pct']:+.2f}%) |",
            f"| **Average Win / Average Loss** | +${trades['avg_win']:,.2f} / -${trades['avg_loss']:,.2f} |",
            f"| **Win / Loss Ratio** | {trades['win_loss_ratio']} |",
            f"| **Max Consecutive Wins / Losses** | {trades['max_consecutive_wins']} / {trades['max_consecutive_losses']} |",
            f"| **Avg Holding Duration** | {trades['avg_holding_days']} days |"
        ])

    return "\n".join(lines)
