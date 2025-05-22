import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:shared_preferences/shared_preferences.dart';

// 알림 플러그인 전역 선언
final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
FlutterLocalNotificationsPlugin();

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initNotifications(); // 알림 초기화
  runApp(const MyApp());
}

Future<void> initNotifications() async {
  const initializationSettingsAndroid =
  AndroidInitializationSettings('@mipmap/ic_launcher');
  const initializationSettings =
  InitializationSettings(android: initializationSettingsAndroid);

  await flutterLocalNotificationsPlugin.initialize(initializationSettings);
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: FirstPage(),
    );
  }
}

class FirstPage extends StatefulWidget {
  const FirstPage({Key? key}) : super(key: key);

  @override
  _FirstPageState createState() => _FirstPageState();
}

class _FirstPageState extends State<FirstPage> {
  double pressure = 0.0; // 기본값 0, 서버에서 받아오면 업데이트
  double minPressure = 45.0;
  double maxPressure = 55.0;

  @override
  void initState() {
    super.initState();
    loadAlarmSettings();
  }

  Future<void> loadAlarmSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      minPressure = prefs.getDouble('minPressure') ?? 45.0;
      maxPressure = prefs.getDouble('maxPressure') ?? 55.0;
    });
  }

  String getStatus(double value) {
    if (value < minPressure) {
      return '공기압 낮음';
    } else if (value > maxPressure) {
      return '공기압 높음';
    } else {
      return '공기압 정상';
    }
  }

  Color getStatusColor(String status) {
    switch (status) {
      case '공기압 낮음':
        return Colors.redAccent;
      case '공기압 높음':
        return Colors.orangeAccent;
      default:
        return Colors.green;
    }
  }

  IconData getStatusIcon(String status) {
    switch (status) {
      case '공기압 낮음':
        return Icons.error;
      case '공기압 높음':
        return Icons.warning;
      default:
        return Icons.check_circle;
    }
  }

  void updatePressure(double newPressure) {
    setState(() {
      pressure = newPressure;
    });
  }

  @override
  Widget build(BuildContext context) {
    final status = getStatus(pressure);
    final statusColor = getStatusColor(status);
    final statusIcon = getStatusIcon(status);

    return Scaffold(
      backgroundColor: Colors.black45,
      appBar: AppBar(
        title: const Text(
          '실시간 타이어 공기압 시스템',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 24),
        ),
        backgroundColor: Colors.blueAccent,
        leading: Builder(
          builder: (context) => IconButton(
            icon: const Icon(Icons.menu),
            onPressed: () => Scaffold.of(context).openDrawer(),
          ),
        ),
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            const DrawerHeader(
              decoration: BoxDecoration(color: Colors.blueAccent),
              child: Text(
                '메뉴',
                style: TextStyle(color: Colors.white, fontSize: 24),
              ),
            ),
            ListTile(
              leading: const Icon(Icons.notifications),
              title: const Text('알림설정'),
              onTap: () {
                Navigator.push(context,
                    MaterialPageRoute(builder: (context) => AlarmPage()));
              },
            ),
            ListTile(
              leading: const Icon(Icons.settings),
              title: const Text('IP 설정'),
              onTap: () async {
                final result = await Navigator.push<double>(
                    context,
                    MaterialPageRoute(builder: (context) => FlaskPage()));
                if (result != null) {
                  updatePressure(result);
                }
              },
            ),
          ],
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 40),
        child: Card(
          elevation: 8,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(
                  Icons.pedal_bike,
                  size: 150,
                  color: Colors.black,
                ),
                const SizedBox(height: 20),
                const Text(
                  '내 자전거의 공기압',
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w600,
                    color: Colors.black87,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  '${pressure.toStringAsFixed(2)} psi',
                  style: const TextStyle(
                    fontSize: 48,
                    fontWeight: FontWeight.bold,
                    color: Colors.blueAccent,
                  ),
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(statusIcon, color: statusColor, size: 28),
                    const SizedBox(width: 8),
                    Text(
                      status,
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: statusColor,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class AlarmPage extends StatefulWidget {
  const AlarmPage({Key? key}) : super(key: key);

  @override
  _AlarmPageState createState() => _AlarmPageState();
}

class _AlarmPageState extends State<AlarmPage> {
  double minPressure = 45.0;
  double maxPressure = 55.0;
  bool _isPushEnabled = true; // 푸시 알림 기본 켜짐

  @override
  void initState() {
    super.initState();
    loadAlarmSettings();
  }

  Future<void> loadAlarmSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      minPressure = prefs.getDouble('minPressure') ?? 45.0;
      maxPressure = prefs.getDouble('maxPressure') ?? 55.0;
      _isPushEnabled = prefs.getBool('push_enabled') ?? true;
    });
  }

  Future<void> saveAlarmSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble('minPressure', minPressure);
    await prefs.setDouble('maxPressure', maxPressure);
    await prefs.setBool('push_enabled', _isPushEnabled);
    ScaffoldMessenger.of(context)
        .showSnackBar(const SnackBar(content: Text('알람 설정이 저장되었습니다.')));
  }

  void _togglePushSetting(bool value) {
    setState(() {
      _isPushEnabled = value;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('알람 설정'),
        backgroundColor: Colors.blue,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            const Text('공기압 알람 임계값 설정',
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            const SizedBox(height: 30),
            const Text('최소 공기압 (이하일 때 알람)', style: TextStyle(fontSize: 18)),
            Slider(
              value: minPressure,
              min: 20,
              max: 60,
              divisions: 40,
              label: '${minPressure.toStringAsFixed(1)} psi',
              onChanged: (value) {
                setState(() {
                  minPressure = value;
                });
              },
            ),
            Text('${minPressure.toStringAsFixed(1)} psi',
                style: const TextStyle(fontSize: 16, color: Colors.blueAccent)),
            const SizedBox(height: 40),
            const Text('최대 공기압 (초과일 때 알람)', style: TextStyle(fontSize: 18)),
            Slider(
              value: maxPressure,
              min: 40,
              max: 80,
              divisions: 40,
              label: '${maxPressure.toStringAsFixed(1)} psi',
              onChanged: (value) {
                setState(() {
                  maxPressure = value;
                });
              },
            ),
            Text('${maxPressure.toStringAsFixed(1)} psi',
                style: const TextStyle(fontSize: 16, color: Colors.blueAccent)),
            const SizedBox(height: 40),
            ListTile(
              title: const Text('푸시 알림 켜기/끄기', style: TextStyle(fontSize: 18)),
              trailing: Switch(
                value: _isPushEnabled,
                onChanged: (value) {
                  _togglePushSetting(value);
                },
              ),
            ),
            const SizedBox(height: 40),
            ElevatedButton(
              onPressed: saveAlarmSettings,
              child: const Text('저장'),
            ),
          ],
        ),
      ),
    );
  }
}

class FlaskPage extends StatefulWidget {
  const FlaskPage({Key? key}) : super(key: key);

  @override
  _FlaskPageState createState() => _FlaskPageState();
}

class _FlaskPageState extends State<FlaskPage> {
  final TextEditingController _ipController = TextEditingController();
  String _responseMessage = '';

  Future<void> fetchData() async {
    final ip = _ipController.text.trim();
    if (ip.isEmpty) {
      setState(() {
        _responseMessage = 'IP를 입력하세요.';
      });
      return;
    }

    final url = 'http://$ip:5000/api/data';
    try {
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        // 서버에서 'pressure' 필드로 공기압 받아옴
        double pressure = (data['pressure'] as num).toDouble();

        setState(() {
          _responseMessage = '현재 공기압: ${pressure.toStringAsFixed(2)} psi';
        });

        // 첫 페이지로 공기압 값 전달하며 닫기
        Navigator.pop(context, pressure);
      } else {
        setState(() {
          _responseMessage = '서버 오류: ${response.statusCode}';
        });
      }
    } catch (e) {
      setState(() {
        _responseMessage = '연결 실패: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('IP 설정 페이지'),
        backgroundColor: Colors.blue,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text('Flask 서버 IP를 입력하세요', style: TextStyle(fontSize: 20)),
            const SizedBox(height: 12),
            TextField(
              controller: _ipController,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: '예: 192.168.0.101',
              ),
              keyboardType: TextInputType.text,
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: fetchData,
              child: const Text('서버에 요청하기'),
            ),
            const SizedBox(height: 20),
            Text(
              _responseMessage,
              style: const TextStyle(fontSize: 18, color: Colors.black87),
            ),
          ],
        ),
      ),
    );
  }
}
