/// Broker Execution Status & Master Boolean Model
class BrokerStatusModel {
  final bool autoExecutionEnabled;
  final String brokerName;
  final bool isPaperTrading;
  final double allocatedCapital;
  final double riskPerTradePct;
  final bool circuitBreakerTripped;
  final String? circuitBreakerReason;

  BrokerStatusModel({
    required this.autoExecutionEnabled,
    required this.brokerName,
    required this.isPaperTrading,
    required this.allocatedCapital,
    required this.riskPerTradePct,
    required this.circuitBreakerTripped,
    this.circuitBreakerReason,
  });

  factory BrokerStatusModel.fromJson(Map<String, dynamic> json) {
    return BrokerStatusModel(
      autoExecutionEnabled: json['auto_execution_enabled'] ?? false,
      brokerName: json['broker_name'] ?? 'AngelOneConnector',
      isPaperTrading: json['is_paper_trading'] ?? true,
      allocatedCapital: (json['allocated_capital'] as num?)?.toDouble() ?? 10000.0,
      riskPerTradePct: (json['risk_per_trade_pct'] as num?)?.toDouble() ?? 2.0,
      circuitBreakerTripped: json['circuit_breaker_tripped'] ?? false,
      circuitBreakerReason: json['circuit_breaker_reason'],
    );
  }
}

class OrderExecutionResult {
  final String status;
  final String message;
  final bool? autoExecutionEnabled;
  final Map<String, dynamic>? order;

  OrderExecutionResult({
    required this.status,
    required this.message,
    this.autoExecutionEnabled,
    this.order,
  });

  factory OrderExecutionResult.fromJson(Map<String, dynamic> json) {
    return OrderExecutionResult(
      status: json['status'] ?? 'unknown',
      message: json['message'] ?? '',
      autoExecutionEnabled: json['auto_execution_enabled'],
      order: json['order'],
    );
  }
}
