import 'package:flutter/material.dart';
import '../services/api_service.dart';

class AutoExecutionToggleWidget extends StatefulWidget {
  final ApiService apiService;
  final String currentSymbol;

  const AutoExecutionToggleWidget({
    Key? key,
    required this.apiService,
    required this.currentSymbol,
  }) : super(key: key);

  @override
  State<AutoExecutionToggleWidget> createState() => _AutoExecutionToggleWidgetState();
}

class _AutoExecutionToggleWidgetState extends State<AutoExecutionToggleWidget> {
  bool _isEnabled = false;
  bool _isLoading = false;
  String _selectedBroker = 'angel_one';
  String? _noticeMessage;
  Color _noticeColor = Colors.grey;

  @override
  void initState() {
    super.initState();
    _fetchStatus();
  }

  Future<void> _fetchStatus() async {
    try {
      final status = await widget.apiService.getExecutionStatus();
      if (mounted) {
        setState(() {
          _isEnabled = status.autoExecutionEnabled;
        });
      }
    } catch (_) {}
  }

  Future<void> _toggleSwitch(bool val) async {
    setState(() => _isLoading = true);
    try {
      final res = await widget.apiService.toggleAutoExecution(
        enabled: val,
        broker: _selectedBroker,
        paperTrading: true,
      );
      if (mounted) {
        setState(() {
          _isEnabled = res['auto_execution_enabled'] ?? false;
          _noticeMessage = _isEnabled
              ? '⚡ Auto-Execution is ON (True): Orders will route to $_selectedBroker'
              : '🛡️ Auto-Execution is OFF (False): Live trades safely locked.';
          _noticeColor = _isEnabled ? const Color(0xFF10B981) : const Color(0xFF94A3B8);
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _noticeMessage = 'Failed to toggle: $e';
          _noticeColor = const Color(0xFFF43F5E);
        });
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _executeNow() async {
    setState(() => _isLoading = true);
    try {
      final res = await widget.apiService.executeOrderNow(widget.currentSymbol);
      if (mounted) {
        setState(() {
          if (res.status == 'blocked') {
            _noticeMessage = '⛔ Blocked: Master Switch is OFF. Toggle to ON first.';
            _noticeColor = const Color(0xFFF43F5E);
          } else if (res.status == 'executed') {
            final o = res.order;
            _noticeMessage = '✅ Order #${o?['order_id']}: ${o?['side']} ${o?['quantity']} ${o?['symbol']} @ \$${o?['price']}';
            _noticeColor = const Color(0xFF10B981);
          } else {
            _noticeMessage = 'ℹ️ ${res.message}';
            _noticeColor = const Color(0xFF06B6D4);
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _noticeMessage = 'Order execution error: $e';
          _noticeColor = const Color(0xFFF43F5E);
        });
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A).withOpacity(0.9),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Master Auto-Execution Boolean',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Colors.white),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _isEnabled ? 'ACTIVE: Automated trades permitted' : 'SAFE: All broker trades locked',
                      style: TextStyle(
                        fontSize: 11,
                        color: _isEnabled ? const Color(0xFF10B981) : const Color(0xFF94A3B8),
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
              Switch(
                value: _isEnabled,
                onChanged: _isLoading ? null : _toggleSwitch,
                activeColor: const Color(0xFF10B981),
                activeTrackColor: const Color(0xFF10B981).withOpacity(0.3),
                inactiveThumbColor: const Color(0xFFF43F5E),
                inactiveTrackColor: const Color(0xFFF43F5E).withOpacity(0.2),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.3),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.white.withOpacity(0.1)),
                  ),
                  child: DropdownButtonHideUnderline(
                    child: DropdownButton<String>(
                      value: _selectedBroker,
                      dropdownColor: const Color(0xFF1E293B),
                      isExpanded: true,
                      style: const TextStyle(fontSize: 12, color: Colors.white),
                      items: const [
                        DropdownMenuItem(value: 'angel_one', child: Text('Angel One (NSE/BSE)')),
                        DropdownMenuItem(value: 'indmoney', child: Text('INDmoney (US Stocks)')),
                      ],
                      onChanged: (val) {
                        if (val != null) setState(() => _selectedBroker = val);
                      },
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              ElevatedButton.icon(
                onPressed: _isLoading ? null : _executeNow,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF10B981),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
                icon: const Icon(Icons.flash_on, size: 16),
                label: const Text('Execute Now', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
              ),
            ],
          ),
          if (_noticeMessage != null) ...[
            const SizedBox(height: 10),
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: _noticeColor.withOpacity(0.12),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: _noticeColor.withOpacity(0.4)),
              ),
              child: Text(
                _noticeMessage!,
                style: TextStyle(color: _noticeColor, fontSize: 11, fontFamily: 'monospace'),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
