/// Backtest Simulation Results Model
class BacktestModel {
  final String symbol;
  final String strategy;
  final double initialCapital;
  final double finalEquity;
  final double totalReturnPct;
  final double benchmarkReturnPct;
  final double sharpeRatio;
  final double sortinoRatio;
  final double maxDrawdownPct;
  final double benchmarkDrawdownPct;
  final double winRatePct;
  final int totalTrades;
  final int winningTrades;
  final int losingTrades;
  final double profitFactor;
  final List<EquityCurvePoint> equityCurve;
  final List<TradeRecord> recentTrades;

  BacktestModel({
    required this.symbol,
    required this.strategy,
    required this.initialCapital,
    required this.finalEquity,
    required this.totalReturnPct,
    required this.benchmarkReturnPct,
    required this.sharpeRatio,
    required this.sortinoRatio,
    required this.maxDrawdownPct,
    required this.benchmarkDrawdownPct,
    required this.winRatePct,
    required this.totalTrades,
    required this.winningTrades,
    required this.losingTrades,
    required this.profitFactor,
    required this.equityCurve,
    required this.recentTrades,
  });

  factory BacktestModel.fromJson(Map<String, dynamic> json) {
    return BacktestModel(
      symbol: json['symbol'] ?? '',
      strategy: json['strategy'] ?? '',
      initialCapital: (json['initial_capital'] as num?)?.toDouble() ?? 10000.0,
      finalEquity: (json['final_equity'] as num?)?.toDouble() ?? 10000.0,
      totalReturnPct: (json['total_return_pct'] as num?)?.toDouble() ?? 0.0,
      benchmarkReturnPct: (json['benchmark_return_pct'] as num?)?.toDouble() ?? 0.0,
      sharpeRatio: (json['sharpe_ratio'] as num?)?.toDouble() ?? 0.0,
      sortinoRatio: (json['sortino_ratio'] as num?)?.toDouble() ?? 0.0,
      maxDrawdownPct: (json['max_drawdown_pct'] as num?)?.toDouble() ?? 0.0,
      benchmarkDrawdownPct: (json['benchmark_drawdown_pct'] as num?)?.toDouble() ?? 0.0,
      winRatePct: (json['win_rate_pct'] as num?)?.toDouble() ?? 0.0,
      totalTrades: (json['total_trades'] as num?)?.toInt() ?? 0,
      winningTrades: (json['winning_trades'] as num?)?.toInt() ?? 0,
      losingTrades: (json['losing_trades'] as num?)?.toInt() ?? 0,
      profitFactor: (json['profit_factor'] as num?)?.toDouble() ?? 0.0,
      equityCurve: (json['equity_curve'] as List<dynamic>?)
              ?.map((e) => EquityCurvePoint.fromJson(e))
              .toList() ??
          [],
      recentTrades: (json['recent_trades'] as List<dynamic>?)
              ?.map((e) => TradeRecord.fromJson(e))
              .toList() ??
          [],
    );
  }
}

class EquityCurvePoint {
  final String date;
  final double strategyEquity;
  final double benchmarkEquity;

  EquityCurvePoint({
    required this.date,
    required this.strategyEquity,
    required this.benchmarkEquity,
  });

  factory EquityCurvePoint.fromJson(Map<String, dynamic> json) {
    return EquityCurvePoint(
      date: json['date'] ?? '',
      strategyEquity: (json['strategy_equity'] as num?)?.toDouble() ?? 0.0,
      benchmarkEquity: (json['benchmark_equity'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class TradeRecord {
  final String entryDate;
  final String exitDate;
  final String action;
  final double entryPrice;
  final double exitPrice;
  final double pnl;
  final double returnPct;
  final String exitReason;

  TradeRecord({
    required this.entryDate,
    required this.exitDate,
    required this.action,
    required this.entryPrice,
    required this.exitPrice,
    required this.pnl,
    required this.returnPct,
    required this.exitReason,
  });

  factory TradeRecord.fromJson(Map<String, dynamic> json) {
    return TradeRecord(
      entryDate: json['entry_date'] ?? '',
      exitDate: json['exit_date'] ?? '',
      action: json['action'] ?? 'BUY',
      entryPrice: (json['entry_price'] as num?)?.toDouble() ?? 0.0,
      exitPrice: (json['exit_price'] as num?)?.toDouble() ?? 0.0,
      pnl: (json['pnl'] as num?)?.toDouble() ?? 0.0,
      returnPct: (json['return_pct'] as num?)?.toDouble() ?? 0.0,
      exitReason: json['exit_reason'] ?? '',
    );
  }
}
