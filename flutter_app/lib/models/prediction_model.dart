/// StockMarketAI: Typed Dart Model for 3-Level Predictions
class PredictionModel {
  final String symbol;
  final String predictionDate;
  final CurrentMarket market;
  final Level1Direction level1;
  final Level2PriceTargets level2;
  final Level3TradingSignal level3;
  final SentimentData sentiment;

  PredictionModel({
    required this.symbol,
    required this.predictionDate,
    required this.market,
    required this.level1,
    required this.level2,
    required this.level3,
    required this.sentiment,
  });

  factory PredictionModel.fromJson(Map<String, dynamic> json) {
    return PredictionModel(
      symbol: json['symbol'] ?? '',
      predictionDate: json['prediction_date'] ?? '',
      market: CurrentMarket.fromJson(json['current_market'] ?? {}),
      level1: Level1Direction.fromJson(json['level1_direction'] ?? {}),
      level2: Level2PriceTargets.fromJson(json['level2_price_targets'] ?? {}),
      level3: Level3TradingSignal.fromJson(json['level3_trading_signal'] ?? {}),
      sentiment: SentimentData.fromJson(json['sentiment'] ?? {}),
    );
  }
}

class CurrentMarket {
  final double open;
  final double high;
  final double low;
  final double close;
  final int volume;
  final double atr;
  final double volatility20;
  final double rsi14;
  final double macd;
  final double macdSignal;
  final double ma20;
  final double ma50;
  final double? vwap;
  final double? vwapDistance;

  CurrentMarket({
    required this.open,
    required this.high,
    required this.low,
    required this.close,
    required this.volume,
    required this.atr,
    required this.volatility20,
    required this.rsi14,
    required this.macd,
    required this.macdSignal,
    required this.ma20,
    required this.ma50,
    this.vwap,
    this.vwapDistance,
  });

  factory CurrentMarket.fromJson(Map<String, dynamic> json) {
    return CurrentMarket(
      open: (json['open'] as num?)?.toDouble() ?? 0.0,
      high: (json['high'] as num?)?.toDouble() ?? 0.0,
      low: (json['low'] as num?)?.toDouble() ?? 0.0,
      close: (json['close'] as num?)?.toDouble() ?? 0.0,
      volume: (json['volume'] as num?)?.toInt() ?? 0,
      atr: (json['atr'] as num?)?.toDouble() ?? 0.0,
      volatility20: (json['volatility_20'] as num?)?.toDouble() ?? 0.0,
      rsi14: (json['rsi_14'] as num?)?.toDouble() ?? 50.0,
      macd: (json['macd'] as num?)?.toDouble() ?? 0.0,
      macdSignal: (json['macd_signal'] as num?)?.toDouble() ?? 0.0,
      ma20: (json['ma_20'] as num?)?.toDouble() ?? 0.0,
      ma50: (json['ma_50'] as num?)?.toDouble() ?? 0.0,
      vwap: (json['vwap'] as num?)?.toDouble(),
      vwapDistance: (json['vwap_distance'] as num?)?.toDouble(),
    );
  }
}

class Level1Direction {
  final String prediction;
  final double probabilityUp;
  final double probabilityPct;
  final double confidence;

  Level1Direction({
    required this.prediction,
    required this.probabilityUp,
    required this.probabilityPct,
    required this.confidence,
  });

  factory Level1Direction.fromJson(Map<String, dynamic> json) {
    return Level1Direction(
      prediction: json['prediction'] ?? 'UP',
      probabilityUp: (json['probability_up'] as num?)?.toDouble() ?? 0.5,
      probabilityPct: (json['probability_pct'] as num?)?.toDouble() ?? 50.0,
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.5,
    );
  }
}

class Level2PriceTargets {
  final double tomorrowOpen;
  final double tomorrowHigh;
  final double tomorrowLow;
  final double tomorrowClose;
  final double expectedChange;
  final double expectedChangePct;
  final double intervalLower;
  final double intervalUpper;

  Level2PriceTargets({
    required this.tomorrowOpen,
    required this.tomorrowHigh,
    required this.tomorrowLow,
    required this.tomorrowClose,
    required this.expectedChange,
    required this.expectedChangePct,
    required this.intervalLower,
    required this.intervalUpper,
  });

  factory Level2PriceTargets.fromJson(Map<String, dynamic> json) {
    return Level2PriceTargets(
      tomorrowOpen: (json['tomorrow_open'] as num?)?.toDouble() ?? 0.0,
      tomorrowHigh: (json['tomorrow_high'] as num?)?.toDouble() ?? 0.0,
      tomorrowLow: (json['tomorrow_low'] as num?)?.toDouble() ?? 0.0,
      tomorrowClose: (json['tomorrow_close'] as num?)?.toDouble() ?? 0.0,
      expectedChange: (json['expected_change'] as num?)?.toDouble() ?? 0.0,
      expectedChangePct: (json['expected_change_pct'] as num?)?.toDouble() ?? 0.0,
      intervalLower: (json['interval_lower'] as num?)?.toDouble() ?? 0.0,
      intervalUpper: (json['interval_upper'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class Level3TradingSignal {
  final String signal;
  final double confidence;
  final double confidencePct;
  final String recommendedAction;
  final double? stopLossPrice;
  final double? takeProfitPrice;
  final double riskRewardRatio;

  Level3TradingSignal({
    required this.signal,
    required this.confidence,
    required this.confidencePct,
    required this.recommendedAction,
    this.stopLossPrice,
    this.takeProfitPrice,
    required this.riskRewardRatio,
  });

  factory Level3TradingSignal.fromJson(Map<String, dynamic> json) {
    return Level3TradingSignal(
      signal: json['signal'] ?? 'HOLD',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.5,
      confidencePct: (json['confidence_pct'] as num?)?.toDouble() ?? 50.0,
      recommendedAction: json['recommended_action'] ?? '',
      stopLossPrice: (json['stop_loss_price'] as num?)?.toDouble(),
      takeProfitPrice: (json['take_profit_price'] as num?)?.toDouble(),
      riskRewardRatio: (json['risk_reward_ratio'] as num?)?.toDouble() ?? 1.0,
    );
  }
}

class SentimentData {
  final double score;
  final String sentimentLabel;
  final int headlineCount;

  SentimentData({
    required this.score,
    required this.sentimentLabel,
    required this.headlineCount,
  });

  factory SentimentData.fromJson(Map<String, dynamic> json) {
    return SentimentData(
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
      sentimentLabel: json['sentiment_label'] ?? 'NEUTRAL',
      headlineCount: (json['headline_count'] as num?)?.toInt() ?? 0,
    );
  }
}
