import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../services/api_service.dart';
import '../models/backtest_model.dart';

class BacktestScreen extends StatefulWidget {
  final ApiService apiService;

  const BacktestScreen({Key? key, required this.apiService}) : super(key: key);

  @override
  State<BacktestScreen> createState() => _BacktestScreenState();
}

class _BacktestScreenState extends State<BacktestScreen> {
  final TextEditingController _symbolController = TextEditingController(text: 'MSFT');
  final TextEditingController _capitalController = TextEditingController(text: '10000');
  String _selectedStrategy = 'enhanced';
  bool _useAtr = true;
  bool _isLoading = false;
  String? _errorMessage;
  BacktestModel? _backtest;

  @override
  void initState() {
    super.initState();
    _runBacktest();
  }

  Future<void> _runBacktest() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final res = await widget.apiService.runBacktest(
        symbol: _symbolController.text.trim().toUpperCase(),
        strategy: _selectedStrategy,
        capital: double.tryParse(_capitalController.text) ?? 10000.0,
        useAtr: _useAtr,
      );
      if (mounted) setState(() => _backtest = res);
    } catch (e) {
      if (mounted) setState(() => _errorMessage = e.toString().replaceAll('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF080C14),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Controls Card
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFF0F172A),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.white.withOpacity(0.08)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('STRATEGY CONFIGURATION', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Color(0xFF94A3B8))),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _symbolController,
                          style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.bold),
                          decoration: InputDecoration(
                            labelText: 'Symbol',
                            labelStyle: const TextStyle(color: Color(0xFF64748B), fontSize: 11),
                            border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                            contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: TextField(
                          controller: _capitalController,
                          keyboardType: TextInputType.number,
                          style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.bold),
                          decoration: InputDecoration(
                            labelText: 'Capital (\$)',
                            labelStyle: const TextStyle(color: Color(0xFF64748B), fontSize: 11),
                            border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                            contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  DropdownButtonFormField<String>(
                    value: _selectedStrategy,
                    dropdownColor: const Color(0xFF1E293B),
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                    decoration: InputDecoration(
                      labelText: 'Trading Strategy',
                      labelStyle: const TextStyle(color: Color(0xFF64748B), fontSize: 11),
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                    ),
                    items: const [
                      DropdownMenuItem(value: 'enhanced', child: Text('Enhanced Hybrid (ML + Dynamic ATR)')),
                      DropdownMenuItem(value: 'ml_direction', child: Text('Pure ML Directional Signal')),
                      DropdownMenuItem(value: 'momentum', child: Text('RSI/MACD Trend Momentum')),
                      DropdownMenuItem(value: 'mean_reversion', child: Text('Bollinger Mean Reversion')),
                    ],
                    onChanged: (val) {
                      if (val != null) setState(() => _selectedStrategy = val);
                    },
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: _isLoading ? null : _runBacktest,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF6366F1),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      ),
                      child: _isLoading
                          ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                          : const Text('Execute Historical Backtest', style: TextStyle(fontWeight: FontWeight.bold)),
                    ),
                  ),
                ],
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
                child: Text(_errorMessage!, style: const TextStyle(color: Color(0xFFF43F5E), fontSize: 12)),
              ),

            if (_backtest != null) ...[
              // Summary Metrics Row
              Row(
                children: [
                  Expanded(
                    child: _buildMetricTile(
                      'Total Return',
                      '${_backtest!.totalReturnPct >= 0 ? "+" : ""}${_backtest!.totalReturnPct.toStringAsFixed(2)}%',
                      'Bench: +${_backtest!.benchmarkReturnPct.toStringAsFixed(1)}%',
                      _backtest!.totalReturnPct >= 0 ? const Color(0xFF10B981) : const Color(0xFFF43F5E),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: _buildMetricTile(
                      'Sharpe Ratio',
                      _backtest!.sharpeRatio.toStringAsFixed(2),
                      'Sortino: ${_backtest!.sortinoRatio.toStringAsFixed(2)}',
                      const Color(0xFF06B6D4),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: _buildMetricTile(
                      'Max Drawdown',
                      '${_backtest!.maxDrawdownPct.toStringAsFixed(2)}%',
                      'Bench: ${_backtest!.benchmarkDrawdownPct.toStringAsFixed(1)}%',
                      const Color(0xFFF59E0B),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: _buildMetricTile(
                      'Win Rate',
                      '${_backtest!.winRatePct.toStringAsFixed(1)}%',
                      '${_backtest!.winningTrades}W / ${_backtest!.losingTrades}L',
                      Colors.white,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),

              // Equity Curve Chart (fl_chart)
              Container(
                height: 240,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: const Color(0xFF0F172A),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.white.withOpacity(0.08)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('EQUITY CURVE vs BENCHMARK', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Color(0xFF94A3B8))),
                    const SizedBox(height: 12),
                    Expanded(
                      child: LineChart(
                        LineChartData(
                          gridData: FlGridData(show: true, drawVerticalLine: false, getDrawingHorizontalLine: (_) => FlLine(color: Colors.white10, strokeWidth: 1)),
                          titlesData: const FlTitlesData(show: false),
                          borderData: FlBorderData(show: false),
                          lineBarsData: [
                            // Strategy Curve (Green)
                            LineChartBarData(
                              spots: _backtest!.equityCurve.asMap().entries.map((e) => FlSpot(e.key.toDouble(), e.value.strategyEquity)).toList(),
                              isCurved: true,
                              color: const Color(0xFF10B981),
                              barWidth: 2.5,
                              dotData: const FlDotData(show: false),
                              belowBarData: BarAreaData(show: true, color: const Color(0xFF10B981).withOpacity(0.08)),
                            ),
                            // Benchmark Curve (Dotted Grey)
                            LineChartBarData(
                              spots: _backtest!.equityCurve.asMap().entries.map((e) => FlSpot(e.key.toDouble(), e.value.benchmarkEquity)).toList(),
                              isCurved: true,
                              color: const Color(0xFF64748B),
                              barWidth: 1.5,
                              dotData: const FlDotData(show: false),
                              dashArray: [4, 4],
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildMetricTile(String label, String value, String sub, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontSize: 10, color: Color(0xFF64748B))),
          const SizedBox(height: 4),
          Text(value, style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: color, fontFamily: 'monospace')),
          const SizedBox(height: 2),
          Text(sub, style: const TextStyle(fontSize: 10, color: Color(0xFF94A3B8))),
        ],
      ),
    );
  }
}
