import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../models/prediction_model.dart';
import '../models/backtest_model.dart';
import '../models/broker_model.dart';

class ApiService {
  static const String _defaultUrl = 'http://10.0.2.2:8000'; // Default for Android emulator, or http://localhost:8000 for iOS/Mac/Windows
  String _baseUrl = _defaultUrl;

  String get baseUrl => _baseUrl;

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _baseUrl = prefs.getString('api_base_url') ?? _defaultUrl;
  }

  Future<void> setBaseUrl(String url) async {
    _baseUrl = url.endsWith('/') ? url.substring(0, url.length - 1) : url;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('api_base_url', _baseUrl);
  }

  /// 1. Health Check
  Future<Map<String, dynamic>> checkHealth() async {
    final res = await http.get(Uri.parse('$_baseUrl/api/v1/health')).timeout(const Duration(seconds: 5));
    if (res.statusCode == 200) {
      return jsonDecode(res.body);
    }
    throw Exception('Server returned status: ${res.statusCode}');
  }

  /// 2. Get 3-Level Prediction (Auto-resolves company names like "SRM Energy Ltd" to tickers)
  Future<PredictionModel> getPrediction(String symbol, {bool useSentiment = true}) async {
    final url = '$_baseUrl/api/v1/predict/${Uri.encodeComponent(symbol.trim())}?use_sentiment=$useSentiment';
    final res = await http.get(Uri.parse(url)).timeout(const Duration(seconds: 15));
    if (res.statusCode == 200) {
      return PredictionModel.fromJson(jsonDecode(res.body));
    }
    final err = jsonDecode(res.body);
    throw Exception(err['detail'] ?? 'Prediction failed');
  }

  /// 2.5 Search Indian & US Stocks by Company Name or Ticker
  Future<List<Map<String, dynamic>>> searchStocks(String query) async {
    final url = '$_baseUrl/api/v1/stocks/search?query=${Uri.encodeComponent(query.trim())}&limit=8';
    final res = await http.get(Uri.parse(url)).timeout(const Duration(seconds: 8));
    if (res.statusCode == 200) {
      final List<dynamic> list = jsonDecode(res.body);
      return list.cast<Map<String, dynamic>>();
    }
    return [];
  }

  /// 3. Run Strategy Backtest
  Future<BacktestModel> runBacktest({
    required String symbol,
    String strategy = 'enhanced',
    double capital = 10000.0,
    bool useAtr = true,
  }) async {
    final res = await http.post(
      Uri.parse('$_baseUrl/api/v1/backtest'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'symbol': symbol.toUpperCase(),
        'strategy': strategy,
        'initial_capital': capital,
        'use_atr': useAtr,
      }),
    ).timeout(const Duration(seconds: 25));

    if (res.statusCode == 200) {
      return BacktestModel.fromJson(jsonDecode(res.body));
    }
    final err = jsonDecode(res.body);
    throw Exception(err['detail'] ?? 'Backtest failed');
  }

  /// 4. Scan Watchlist
  Future<List<Map<String, dynamic>>> scanWatchlist(List<String> symbols) async {
    final res = await http.post(
      Uri.parse('$_baseUrl/api/v1/stocks/scan'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'symbols': symbols}),
    ).timeout(const Duration(seconds: 30));

    if (res.statusCode == 200) {
      final List<dynamic> list = jsonDecode(res.body);
      return list.cast<Map<String, dynamic>>();
    }
    throw Exception('Watchlist scan failed');
  }

  /// 5. Get Master Broker Execution Status
  Future<BrokerStatusModel> getExecutionStatus() async {
    final res = await http.get(Uri.parse('$_baseUrl/api/v1/brokers/execution-status')).timeout(const Duration(seconds: 6));
    if (res.statusCode == 200) {
      return BrokerStatusModel.fromJson(jsonDecode(res.body));
    }
    throw Exception('Failed to get broker status');
  }

  /// 6. Toggle Master Auto-Execution Boolean Switch
  Future<Map<String, dynamic>> toggleAutoExecution({
    required bool enabled,
    String broker = 'angel_one',
    bool paperTrading = true,
  }) async {
    final res = await http.post(
      Uri.parse('$_baseUrl/api/v1/brokers/toggle-execution'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'enabled': enabled,
        'broker': broker,
        'paper_trading': paperTrading,
      }),
    ).timeout(const Duration(seconds: 8));

    if (res.statusCode == 200) {
      return jsonDecode(res.body);
    }
    throw Exception('Failed to toggle auto execution');
  }

  /// 7. Execute AI Order Now (Gated by Boolean Switch)
  Future<OrderExecutionResult> executeOrderNow(String symbol) async {
    final res = await http.post(
      Uri.parse('$_baseUrl/api/v1/brokers/execute-now'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'symbol': symbol.toUpperCase()}),
    ).timeout(const Duration(seconds: 10));

    if (res.statusCode == 200) {
      return OrderExecutionResult.fromJson(jsonDecode(res.body));
    }
    throw Exception('Failed to execute order');
  }
}
