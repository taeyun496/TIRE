import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';

class WebSocketService {
  WebSocketChannel? _channel;
  final String _url = 'ws://your-raspberry-pi-ip:8765'; // 라즈베리파이 IP로 변경 필요

  void connect({
    required Function(Map<String, dynamic>) onMessage,
    required Function(String) onError,
  }) {
    try {
      _channel = WebSocketChannel.connect(Uri.parse(_url));
      
      _channel!.stream.listen(
        (message) {
          try {
            final data = json.decode(message);
            onMessage(data);
          } catch (e) {
            onError('데이터 파싱 오류: $e');
          }
        },
        onError: (error) {
          onError('연결 오류: $error');
        },
        onDone: () {
          onError('연결이 종료되었습니다.');
        },
      );
    } catch (e) {
      onError('연결 실패: $e');
    }
  }

  void disconnect() {
    _channel?.sink.close();
  }

  void sendMessage(String message) {
    _channel?.sink.add(message);
  }
} 