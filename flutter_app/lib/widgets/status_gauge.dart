import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

class StatusGauge extends StatelessWidget {
  final String status;
  final double confidence;

  StatusGauge({
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

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 200,
      child: Stack(
        children: [
          PieChart(
            PieChartData(
              sections: [
                PieChartSectionData(
                  value: confidence * 100,
                  color: _getStatusColor(),
                  title: '${(confidence * 100).toStringAsFixed(1)}%',
                  radius: 100,
                  titleStyle: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                PieChartSectionData(
                  value: (1 - confidence) * 100,
                  color: Colors.grey[200],
                  radius: 100,
                ),
              ],
              sectionsSpace: 0,
              centerSpaceRadius: 40,
            ),
          ),
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  status,
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: _getStatusColor(),
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  '신뢰도',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.grey[600],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
} 