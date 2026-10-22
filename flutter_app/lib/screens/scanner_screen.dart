import 'package:flutter/material.dart';
import '../services/api_service.dart';

class ScannerScreen extends StatefulWidget {
  final ApiService apiService;
  final Function(String) onSelectSymbol;

  const ScannerScreen({
    Key? key,
    required this.apiService,
    required this.onSelectSymbol,
  }) : super(key: key);

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  bool _isLoading = false;
  String? _errorMessage;
  List<Map<String, dynamic>> _results = [];

  final List<String> _defaultWatchlist = ['MSFT', 'AAPL', 'NVDA', 'TSLA', 'AGYS', 'GOOGL', 'RELIANCE.NS'];

  @override
  void initState() {
    super.initState();
    _scanWatchlist();
  }

  Future<void> _scanWatchlist() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final res = await widget.apiService.scanWatchlist(_defaultWatchlist);
      if (mounted) setState(() => _results = res);
    } catch (e) {
      if (mounted) setState(() => _errorMessage = e.toString().replaceAll('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Color _getSignalColor(String signal) {
    if (signal.contains('BUY')) return const Color(0xFF10B981);
    if (signal.contains('SELL')) return const Color(0xFFF43F5E);
    return const Color(0xFFF59E0B);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF080C14),
      body: RefreshIndicator(
        onRefresh: _scanWatchlist,
        color: const Color(0xFF6366F1),
        backgroundColor: const Color(0xFF1E293B),
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('WATCHLIST OPPORTUNITY SCANNER', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Color(0xFF94A3B8))),
                IconButton(
                  icon: _isLoading
                      ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(color: Color(0xFF6366F1), strokeWidth: 2))
                      : const Icon(Icons.refresh, color: Color(0xFF6366F1), size: 20),
                  onPressed: _isLoading ? null : _scanWatchlist,
                ),
              ],
            ),
            const SizedBox(height: 8),

            if (_errorMessage != null)
              Container(
                padding: const EdgeInsets.all(12),
                margin: const EdgeInsets.only(bottom: 12),
                decoration: BoxDecoration(
                  color: const Color(0xFFF43F5E).withOpacity(0.12),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: const Color(0xFFF43F5E).withOpacity(0.4)),
                ),
                child: Text(_errorMessage!, style: const TextStyle(color: Color(0xFFF43F5E), fontSize: 12)),
              ),

            ..._results.map((item) {
              final sym = item['symbol'] ?? '';
              final price = (item['current_price'] as num?)?.toDouble() ?? 0.0;
              final target = (item['predicted_close'] as num?)?.toDouble() ?? 0.0;
              final change = (item['expected_change_pct'] as num?)?.toDouble() ?? 0.0;
              final signal = item['signal'] ?? 'HOLD';
              final conf = (item['confidence'] as num?)?.toDouble() ?? 50.0;
              final isPos = change >= 0;
              final sigColor = _getSignalColor(signal);

              return Container(
                margin: const EdgeInsets.only(bottom: 10),
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: const Color(0xFF0F172A),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.white.withOpacity(0.06)),
                ),
                child: Row(
                  children: [
                    // Symbol & Price
                    Expanded(
                      flex: 2,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(sym, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white)),
                          const SizedBox(height: 2),
                          Text('\$${price.toStringAsFixed(2)}', style: const TextStyle(fontSize: 12, color: Color(0xFF94A3B8), fontFamily: 'monospace')),
                        ],
                      ),
                    ),

                    // Expected Change
                    Expanded(
                      flex: 2,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Target: \$${target.toStringAsFixed(2)}', style: const TextStyle(fontSize: 11, color: Color(0xFF64748B))),
                          const SizedBox(height: 2),
                          Text(
                            '${isPos ? "+" : ""}${change.toStringAsFixed(2)}%',
                            style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: isPos ? const Color(0xFF10B981) : const Color(0xFFF43F5E)),
                          ),
                        ],
                      ),
                    ),

                    // Signal Badge & Action
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: sigColor.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(6),
                            border: Border.all(color: sigColor.withOpacity(0.4)),
                          ),
                          child: Text(
                            '$signal (${conf.round()}%)',
                            style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: sigColor),
                          ),
                        ),
                        const SizedBox(height: 4),
                        GestureDetector(
                          onTap: () => widget.onSelectSymbol(sym),
                          child: const Text('Inspect →', style: TextStyle(fontSize: 11, color: Color(0xFF6366F1), fontWeight: FontWeight.bold)),
                        ),
                      ],
                    ),
                  ],
                ),
              );
            }).toList(),
          ],
        ),
      ),
    );
  }
}
