#!/usr/bin/env python3
"""
StockMarketAI: Predict ANY Stock Symbol
Simple interface to predict tomorrow's OHLC for any stock symbol
"""

import subprocess
import sys

def predict_stock(symbol):
    """Predict tomorrow's OHLC for a given stock symbol using the complete workflow"""
    print(f"🔮 StockMarketAI: Predicting tomorrow's OHLC for {symbol.upper()}")
    print("=" * 60)

    try:
        # Run the complete workflow
        result = subprocess.run([
            sys.executable,
            "complete_workflow_fixed.py",
            symbol.upper()
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ Error predicting {symbol}:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout predicting {symbol} (took too long)")
        return False
    except Exception as e:
        print(f"❌ Failed to predict {symbol}: {e}")
        return False

def main():
    print("🚀 StockMarketAI: Predict ANY Stock Symbol")
    print("=" * 50)
    print("Enter stock symbols to get tomorrow's OHLC prediction")
    print("Type 'quit' or 'exit' to stop")
    print("Examples: AGYS, MSFT, AAPL, TSLA, NVDA")
    print()

    while True:
        try:
            user_input = input("📈 Enter stock symbol: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                print("⚠️  Please enter a stock symbol")
                continue

            # Basic validation - should contain only letters, dots, and hyphens
            if not all(c.isalpha() or c in '.-' for c in user_input):
                print("⚠️  Please enter a valid stock symbol (letters, dots, hyphens only)")
                continue

            predict_stock(user_input)
            print()  # Empty line for readability

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    # If symbol provided as command line argument, predict once and exit
    if len(sys.argv) > 1:
        symbol = sys.argv[1].strip()
        predict_stock(symbol)
    else:
        # Interactive mode
        main()