import 'package:flutter/material.dart';
import '../widgets/tire_status.dart';
import '../widgets/status_gauge.dart';
import '../services/websocket_service.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final WebSocketService _webSocketService = WebSocketService();
  String _tireStatus = '연결 대기 중...';
  double _confidence = 0.0;

  @override
  void initState() {
    super.initState();
    _connectWebSocket();
  }

  void _connectWebSocket() {
    _webSocketService.connect(
      onMessage: (data) {
        setState(() {
          _tireStatus = data['status'] == 'normal' ? '정상' : '마모됨';
          _confidence = data['confidence'];
        });
      },
      onError: (error) {
        setState(() {
          _tireStatus = '연결 오류';
        });
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('타이어 상태 모니터링'),
        actions: [
          IconButton(
            icon: Icon(Icons.history),
            onPressed: () {
              Navigator.pushNamed(context, '/history');
            },
          ),
        ],
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            StatusGauge(
              status: _tireStatus,
              confidence: _confidence,
            ),
            SizedBox(height: 20),
            TireStatus(
              status: _tireStatus,
              confidence: _confidence,
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _connectWebSocket,
              child: Text('재연결'),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _webSocketService.disconnect();
    super.dispose();
  }
} 