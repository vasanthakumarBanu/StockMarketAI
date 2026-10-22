import 'package:flutter/material.dart';
import '../models/prediction_model.dart';

class HeroSignalCard extends StatelessWidget {
  final PredictionModel prediction;

  const HeroSignalCard({Key? key, required this.prediction}) : super(key: key);

  Color _getSignalColor(String signal) {
    if (signal.contains('BUY')) return const Color(0xFF10B981);
    if (signal.contains('SELL')) return const Color(0xFFF43F5E);
    return const Color(0xFFF59E0B);
  }

  @override
  Widget build(BuildContext context) {
    final l3 = prediction.level3;
    final l2 = prediction.level2;
    final m = prediction.market;
    final signalColor = _getSignalColor(l3.signal);
    final isPos = l2.expectedChangePct >= 0;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            const Color(0xFF1E293B).withOpacity(0.9),
            const Color(0xFF0F172A).withOpacity(0.95),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: signalColor.withOpacity(0.35)),
        boxShadow: [
          BoxShadow(
            color: signalColor.withOpacity(0.12),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header Row: Level tag + Signal Title + Confidence
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.08),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: const Text(
                      'LEVEL 3 SIGNAL',
                      style: TextStyle(color: Color(0xFF94A3B8), fontSize: 10, fontWeight: FontWeight.bold),
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    l3.signal,
                    style: TextStyle(
                      color: signalColor,
                      fontSize: 32,
                      fontWeight: FontWeight.w900,
                      letterSpacing: -0.5,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    l3.recommendedAction,
                    style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 12),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.4),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.white.withOpacity(0.08)),
                ),
                child: Column(
                  children: [
                    Text(
                      '${l3.confidencePct.round()}%',
                      style: const TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                        fontFamily: 'monospace',
                      ),
                    ),
                    const Text('Confidence', style: TextStyle(fontSize: 10, color: Color(0xFF64748B))),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          const Divider(color: Color(0x1FFFFFFF), height: 1),
          const SizedBox(height: 16),

          // Price Metrics Grid
          Row(
            children: [
              Expanded(
                child: _buildMetric(
                  'Current Close',
                  '\$${m.close.toStringAsFixed(2)}',
                  null,
                  Colors.white,
                ),
              ),
              Expanded(
                child: _buildMetric(
                  'Tomorrow Target',
                  '\$${l2.tomorrowClose.toStringAsFixed(2)}',
                  '${isPos ? "+" : ""}${l2.expectedChangePct.toStringAsFixed(2)}%',
                  const Color(0xFF06B6D4),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _buildMetric(
                  'Stop Loss (ATR)',
                  l3.stopLossPrice != null ? '\$${l3.stopLossPrice!.toStringAsFixed(2)}' : 'N/A',
                  null,
                  const Color(0xFFF43F5E),
                ),
              ),
              Expanded(
                child: _buildMetric(
                  'Take Profit',
                  l3.takeProfitPrice != null ? '\$${l3.takeProfitPrice!.toStringAsFixed(2)}' : 'N/A',
                  '${l3.riskRewardRatio}:1 R:R',
                  const Color(0xFF10B981),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMetric(String label, String value, String? badge, Color valColor) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 4),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A).withOpacity(0.6),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontSize: 10, color: Color(0xFF64748B))),
          const SizedBox(height: 4),
          Row(
            children: [
              Text(
                value,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                  color: valColor,
                  fontFamily: 'monospace',
                ),
              ),
              if (badge != null) ...[
                const SizedBox(width: 4),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                  decoration: BoxDecoration(
                    color: valColor.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(3),
                  ),
                  child: Text(
                    badge,
                    style: TextStyle(fontSize: 9, fontWeight: FontWeight.bold, color: valColor),
                  ),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }
}
