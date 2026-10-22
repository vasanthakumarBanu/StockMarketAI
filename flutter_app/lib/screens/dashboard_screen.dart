import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../models/prediction_model.dart';
import '../widgets/hero_signal_card.dart';
import '../widgets/auto_execution_toggle.dart';

class DashboardScreen extends StatefulWidget {
  final ApiService apiService;

  const DashboardScreen({Key? key, required this.apiService}) : super(key: key);

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final TextEditingController _symbolController = TextEditingController(text: 'MSFT');
  String _currentSymbol = 'MSFT';
  bool _isLoading = false;
  String? _errorMessage;
  PredictionModel? _prediction;

  final List<String> _popularTickers = ['MSFT', 'AAPL', 'NVDA', 'TSLA', 'AGYS', 'RELIANCE.NS'];

  @override
  void initState() {
    super.initState();
    _loadPrediction(_currentSymbol);
  }

  Future<void> _loadPrediction(String symbol) async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final res = await widget.apiService.getPrediction(symbol);
      if (mounted) {
        setState(() {
          _prediction = res;
          _currentSymbol = symbol;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = e.toString().replaceAll('Exception: ', '');
        });
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF080C14),
      body: RefreshIndicator(
        onRefresh: () => _loadPrediction(_currentSymbol),
        color: const Color(0xFF6366F1),
        backgroundColor: const Color(0xFF1E293B),
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Search Row
              Row(
                children: [
                  Expanded(
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color(0xFF0F172A),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: Colors.white.withOpacity(0.08)),
                      ),
                      child: TextField(
                        controller: _symbolController,
                        style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                        textCapitalization: TextCapitalization.characters,
                        decoration: const InputDecoration(
                          hintText: 'Enter stock symbol (e.g. MSFT, AAPL)...',
                          hintStyle: TextStyle(color: Color(0xFF64748B), fontSize: 13),
                          prefixIcon: Icon(Icons.search, color: Color(0xFF64748B), size: 18),
                          border: InputBorder.none,
                          contentPadding: EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                        ),
                        onSubmitted: (val) {
                          if (val.trim().isNotEmpty) _loadPrediction(val.trim().toUpperCase());
                        },
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  ElevatedButton(
                    onPressed: _isLoading ? null : () => _loadPrediction(_symbolController.text.trim().toUpperCase()),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF6366F1),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                    child: _isLoading
                        ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                        : const Text('Predict', style: TextStyle(fontWeight: FontWeight.bold)),
                  ),
                ],
              ),
              const SizedBox(height: 12),

              // Quick Ticker Chips
              SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: _popularTickers.map((sym) {
                    final isSelected = sym == _currentSymbol;
                    return Padding(
                      padding: const EdgeInsets.only(right: 6),
                      child: ChoiceChip(
                        label: Text(sym, style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: isSelected ? Colors.white : const Color(0xFF94A3B8))),
                        selected: isSelected,
                        selectedColor: const Color(0xFF6366F1).withOpacity(0.3),
                        backgroundColor: const Color(0xFF0F172A),
                        side: BorderSide(color: isSelected ? const Color(0xFF6366F1) : Colors.white.withOpacity(0.08)),
                        onSelected: (_) {
                          _symbolController.text = sym;
                          _loadPrediction(sym);
                        },
                      ),
                    );
                  }).toList(),
                ),
              ),
              const SizedBox(height: 16),

              if (_errorMessage != null)
                Container(
                  padding: const EdgeInsets.all(12),
                  margin: const EdgeInsets.only(bottom: 16),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF43F5E).withOpacity(0.12),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: const Color(0xFFF43F5E).withOpacity(0.4)),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.error_outline, color: Color(0xFFF43F5E), size: 18),
                      const SizedBox(width: 8),
                      Expanded(child: Text(_errorMessage!, style: const TextStyle(color: Color(0xFFF43F5E), fontSize: 12))),
                    ],
                  ),
                ),

              if (_prediction != null) ...[
                // Hero Signal Card
                HeroSignalCard(prediction: _prediction!),
                const SizedBox(height: 16),

                // Master Auto-Execution Boolean Switch Widget
                AutoExecutionToggleWidget(
                  apiService: widget.apiService,
                  currentSymbol: _currentSymbol,
                ),
                const SizedBox(height: 16),

                // Level 1: Direction & Sentiment Row
                Row(
                  children: [
                    Expanded(
                      child: _buildDirectionCard(_prediction!.level1),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: _buildSentimentCard(_prediction!.sentiment),
                    ),
                  ],
                ),
                const SizedBox(height: 16),

                // Level 2: Tomorrow OHLC Target Grid
                _buildOhlcGrid(_prediction!.level2),
                const SizedBox(height: 16),

                // Technical Indicators Context Grid
                _buildIndicatorsGrid(_prediction!.market),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDirectionCard(Level1Direction l1) {
    final isUp = l1.prediction == 'UP';
    final color = isUp ? const Color(0xFF10B981) : const Color(0xFFF43F5E);

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('LEVEL 1 DIRECTION', style: TextStyle(fontSize: 10, color: Color(0xFF64748B), fontWeight: FontWeight.bold)),
          const SizedBox(height: 6),
          Row(
            children: [
              Icon(isUp ? Icons.arrow_upward : Icons.arrow_downward, color: color, size: 20),
              const SizedBox(width: 4),
              Text(l1.prediction, style: TextStyle(color: color, fontSize: 18, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 4),
          Text('${l1.probabilityPct.toStringAsFixed(1)}% Upward Prob', style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 11)),
        ],
      ),
    );
  }

  Widget _buildSentimentCard(SentimentData s) {
    Color sentColor = const Color(0xFF94A3B8);
    if (s.score > 0.1) sentColor = const Color(0xFF10B981);
    if (s.score < -0.1) sentColor = const Color(0xFFF43F5E);

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('VADER NEWS SENTIMENT', style: TextStyle(fontSize: 10, color: Color(0xFF64748B), fontWeight: FontWeight.bold)),
          const SizedBox(height: 6),
          Text(s.sentimentLabel, style: TextStyle(color: sentColor, fontSize: 16, fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          Text('Score: ${s.score >= 0 ? "+" : ""}${s.score.toStringAsFixed(2)} (${s.headlineCount} news)',
              style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 11)),
        ],
      ),
    );
  }

  Widget _buildOhlcGrid(Level2PriceTargets l2) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('LEVEL 2: TOMORROW OHLC TARGETS', style: TextStyle(fontSize: 11, color: Color(0xFF06B6D4), fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _ohlcItem('Open', '\$${l2.tomorrowOpen.toStringAsFixed(2)}'),
              _ohlcItem('High', '\$${l2.tomorrowHigh.toStringAsFixed(2)}', color: const Color(0xFF10B981)),
              _ohlcItem('Low', '\$${l2.tomorrowLow.toStringAsFixed(2)}', color: const Color(0xFFF43F5E)),
              _ohlcItem('Close', '\$${l2.tomorrowClose.toStringAsFixed(2)}', color: const Color(0xFF06B6D4)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _ohlcItem(String title, String val, {Color color = Colors.white}) {
    return Column(
      children: [
        Text(title, style: const TextStyle(fontSize: 11, color: Color(0xFF64748B))),
        const SizedBox(height: 4),
        Text(val, style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: color, fontFamily: 'monospace')),
      ],
    );
  }

  Widget _buildIndicatorsGrid(CurrentMarket m) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('TECHNICAL FACTORS CONTEXT (65+ INDICATORS)', style: TextStyle(fontSize: 11, color: Color(0xFF94A3B8), fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _indicatorItem('RSI (14)', m.rsi14.toStringAsFixed(1)),
              _indicatorItem('MACD', m.macd.toStringAsFixed(2)),
              _indicatorItem('MA 20', '\$${m.ma20.toStringAsFixed(1)}'),
              _indicatorItem('VWAP', m.vwap != null ? '\$${m.vwap!.toStringAsFixed(1)}' : 'N/A'),
              _indicatorItem('ATR', '\$${m.atr.toStringAsFixed(2)}'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _indicatorItem(String label, String value) {
    return Column(
      children: [
        Text(label, style: const TextStyle(fontSize: 10, color: Color(0xFF64748B))),
        const SizedBox(height: 4),
        Text(value, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white, fontFamily: 'monospace')),
      ],
    );
  }
}
