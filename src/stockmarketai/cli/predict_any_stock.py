#!/usr/bin/env python3
"""
StockMarketAI: Predict ANY Stock Symbol CLI
High-performance CLI interface to predict tomorrow's OHLC and AI recommendations for any stock symbol
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.stockmarketai.service import StockMarketAIService


def predict_stock(symbol: str) -> bool:
    """Predict tomorrow's OHLC for a given stock symbol using StockMarketAIService."""
    print(f"\n🔮 StockMarketAI: Analyzing '{symbol.upper()}'...")
    print("=" * 65)

    try:
        service = StockMarketAIService()
        res = service.predict(symbol)

        resolved_sym = res.get("symbol", symbol).upper()
        currency = "₹" if resolved_sym.endswith(".NS") or resolved_sym.endswith(".BO") else "$"
        market = res.get("current_market", {})

        level1 = res.get("level1_direction", {})
        level2 = res.get("level2_price_targets", {})
        level3 = res.get("level3_trading_signal", {})
        sentiment = res.get("sentiment", {})

        print(f"Company:        {res.get('company_name', symbol.upper())}")
        print(f"Ticker:         {res.get('symbol', symbol.upper())} ({res.get('market', 'Equities')})")
        print(f"Latest Close:   {currency}{market.get('close', 0.0):.2f} (Vol: {market.get('volume', 0):,})")
        print(f"Direction AI:   {level1.get('prediction', 'UNKNOWN')} ({level1.get('probability_pct', 0.0):.1f}% UP probability)")
        print(f"Trading Signal: {level3.get('signal', 'HOLD')} (Conf: {level3.get('confidence', 0.0)*100:.1f}%)")
        print(f"Trade Plan:     Entry: {currency}{level3.get('suggested_entry', 0.0):.2f} | Stop Loss: {currency}{level3.get('stop_loss', 0.0):.2f} | Target: {currency}{level3.get('take_profit', 0.0):.2f} | R:R: {level3.get('risk_reward_ratio', 1.0):.1f}")
        print(f"News Sentiment: {sentiment.get('label', 'NEUTRAL')} (Score: {sentiment.get('score', 0.0):+.2f})")
        print("\nPredicted Tomorrow OHLC Prices:")
        for k in ['open', 'high', 'low', 'close']:
            cur_p = market.get(k, 0.0)
            pred_p = level2.get(f'tomorrow_{k}', 0.0)
            chg = pred_p - cur_p
            pct = (chg / cur_p * 100) if cur_p else 0
            print(f"  Tomorrow {k.upper():5s}: {currency}{pred_p:8.2f}  (Chg: {chg:+7.2f}, {pct:+6.2f}%)")

        print("=" * 65)
        return True


    except Exception as e:
        print(f"❌ Failed to predict {symbol}: {e}")
        return False


def main():
    print("🚀 StockMarketAI: Multi-Market Stock Predictor")
    print("=" * 55)
    print("Enter any US or Indian stock symbol or company name")
    print("Type 'quit' or 'exit' to stop")
    print("Examples: MSFT, AAPL, TSLA, RELIANCE, TCS, INFY, SRM Energy")
    print()

    while True:
        try:
            user_input = input("📈 Enter stock symbol or name: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                print("⚠️  Please enter a stock symbol or name")
                continue

            predict_stock(user_input)

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_stock(" ".join(sys.argv[1:]))
    else:
        main()