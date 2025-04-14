import 'package:flutter/material.dart';

class TireStatus extends StatelessWidget {
  final String status;
  final double confidence;

  TireStatus({
    required this.status,
    required this.confidence,
  });

  Color _getStatusColor() {
    switch (status) {
      case '정상':
        return Colors.green;
      case '마모됨':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  IconData _getStatusIcon() {
    switch (status) {
      case '정상':
        return Icons.check_circle;
      case '마모됨':
        return Icons.warning;
      default:
        return Icons.sync;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      child: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  _getStatusIcon(),
                  color: _getStatusColor(),
                  size: 48,
                ),
                SizedBox(width: 16),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '타이어 상태',
                      style: TextStyle(
                        fontSize: 16,
                        color: Colors.grey[600],
                      ),
                    ),
                    Text(
                      status,
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: _getStatusColor(),
                      ),
                    ),
                  ],
                ),
              ],
            ),
            SizedBox(height: 16),
            LinearProgressIndicator(
              value: confidence,
              backgroundColor: Colors.grey[200],
              valueColor: AlwaysStoppedAnimation<Color>(_getStatusColor()),
            ),
            SizedBox(height: 8),
            Text(
              '신뢰도: ${(confidence * 100).toStringAsFixed(1)}%',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[600],
              ),
            ),
          ],
        ),
      ),
    );
  }
} 