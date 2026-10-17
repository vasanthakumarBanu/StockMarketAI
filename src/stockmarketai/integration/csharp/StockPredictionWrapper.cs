// ==============================================================================
// StockMarketAI: C# Platform Wrapper
// Implements Section 2.2 of multi_stock_adaptation_plan.md
// Provides .NET Interop for trading platforms (Angel One, INDmoney, NinjaTrader, etc.)
// ==============================================================================

using System;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using System.Net.Http;
using System.Text;

namespace StockMarketAI.PlatformIntegration
{
    public interface IStockPredictionService
    {
        PredictionResult Predict(string stockSymbol, bool useSentiment = true);
        Task<PredictionResult> PredictAsync(string stockSymbol, bool useSentiment = true);
        BacktestResults Backtest(string stockSymbol, string strategy = "enhanced", double initialCapital = 10000.0);
        SystemStatus GetStatus();
    }

    public class PredictionResult
    {
        [JsonPropertyName("symbol")]
        public string Symbol { get; set; }

        [JsonPropertyName("prediction_date")]
        public string PredictionDate { get; set; }

        [JsonPropertyName("current_market")]
        public MarketData CurrentMarket { get; set; }

        [JsonPropertyName("level1_direction")]
        public DirectionData Level1Direction { get; set; }

        [JsonPropertyName("level2_price_targets")]
        public PriceTargets Level2PriceTargets { get; set; }

        [JsonPropertyName("level3_trading_signal")]
        public TradingSignal Level3TradingSignal { get; set; }
    }

    public class MarketData
    {
        [JsonPropertyName("close")]
        public double Close { get; set; }

        [JsonPropertyName("open")]
        public double Open { get; set; }

        [JsonPropertyName("high")]
        public double High { get; set; }

        [JsonPropertyName("low")]
        public double Low { get; set; }

        [JsonPropertyName("atr")]
        public double ATR { get; set; }

        [JsonPropertyName("rsi_14")]
        public double RSI14 { get; set; }
    }

    public class DirectionData
    {
        [JsonPropertyName("prediction")]
        public string Prediction { get; set; }

        [JsonPropertyName("probability_pct")]
        public double ProbabilityPct { get; set; }

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }
    }

    public class PriceTargets
    {
        [JsonPropertyName("tomorrow_open")]
        public double TomorrowOpen { get; set; }

        [JsonPropertyName("tomorrow_high")]
        public double TomorrowHigh { get; set; }

        [JsonPropertyName("tomorrow_low")]
        public double TomorrowLow { get; set; }

        [JsonPropertyName("tomorrow_close")]
        public double TomorrowClose { get; set; }

        [JsonPropertyName("expected_change_pct")]
        public double ExpectedChangePct { get; set; }
    }

    public class TradingSignal
    {
        [JsonPropertyName("signal")]
        public string Signal { get; set; } // STRONG BUY, BUY, HOLD, SELL, STRONG SELL

        [JsonPropertyName("confidence_pct")]
        public double ConfidencePct { get; set; }

        [JsonPropertyName("stop_loss_price")]
        public double? StopLossPrice { get; set; }

        [JsonPropertyName("take_profit_price")]
        public double? TakeProfitPrice { get; set; }

        [JsonPropertyName("recommended_action")]
        public string RecommendedAction { get; set; }
    }

    public class BacktestResults
    {
        [JsonPropertyName("symbol")]
        public string Symbol { get; set; }

        [JsonPropertyName("final_equity")]
        public double FinalEquity { get; set; }

        [JsonPropertyName("total_return_pct")]
        public double TotalReturnPct { get; set; }

        [JsonPropertyName("sharpe_ratio")]
        public double SharpeRatio { get; set; }

        [JsonPropertyName("max_drawdown_pct")]
        public double MaxDrawdownPct { get; set; }

        [JsonPropertyName("win_rate_pct")]
        public double WinRatePct { get; set; }

        [JsonPropertyName("profit_factor")]
        public double ProfitFactor { get; set; }
    }

    public class SystemStatus
    {
        [JsonPropertyName("status")]
        public string Status { get; set; }

        [JsonPropertyName("version")]
        public string Version { get; set; }
    }

    /// <summary>
    /// Production C# Wrapper that integrates with the StockMarketAI Engine via REST or Native Interop.
    /// </summary>
    public class StockPredictionWrapper : IStockPredictionService, IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public StockPredictionWrapper(string baseUrl = "http://localhost:8000")
        {
            _baseUrl = baseUrl.TrimEnd('/');
            _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        }

        public PredictionResult Predict(string stockSymbol, bool useSentiment = true)
        {
            return PredictAsync(stockSymbol, useSentiment).GetAwaiter().GetResult();
        }

        public async Task<PredictionResult> PredictAsync(string stockSymbol, bool useSentiment = true)
        {
            var url = $"{_baseUrl}/api/v1/predict/{Uri.EscapeDataString(stockSymbol)}?use_sentiment={useSentiment.ToString().ToLower()}";
            var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<PredictionResult>(json);
        }

        public BacktestResults Backtest(string stockSymbol, string strategy = "enhanced", double initialCapital = 10000.0)
        {
            var url = $"{_baseUrl}/api/v1/backtest";
            var payload = JsonSerializer.Serialize(new
            {
                symbol = stockSymbol,
                strategy = strategy,
                initial_capital = initialCapital,
                use_atr = true
            });

            var content = new StringContent(payload, Encoding.UTF8, "application/json");
            var response = _httpClient.PostAsync(url, content).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
            var json = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
            return JsonSerializer.Deserialize<BacktestResults>(json);
        }

        public SystemStatus GetStatus()
        {
            var url = $"{_baseUrl}/api/v1/status";
            var response = _httpClient.GetAsync(url).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
            var json = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
            return JsonSerializer.Deserialize<SystemStatus>(json);
        }

        public void Dispose()
        {
            _httpClient?.Dispose();
        }
    }
}
