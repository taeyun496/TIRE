import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:lottie/lottie.dart';

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
      theme: ThemeData(
        primarySwatch: Colors.indigo,
        scaffoldBackgroundColor: Colors.grey[100],
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.indigo,
          elevation: 0,
        ),
        cardTheme: CardTheme(
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      home: const OnboardingScreen(),
    );
  }
}

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({Key? key}) : super(key: key);

  @override
  _OnboardingScreenState createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _pageController = PageController();
  int _currentPage = 0;

  final List<OnboardingPage> _pages = [
    OnboardingPage(
      title: '타이어 공기압 모니터링',
      description: '실시간으로 자전거 타이어의 공기압을 모니터링하세요.',
      icon: Icons.pedal_bike,
    ),
    OnboardingPage(
      title: '스마트 알림',
      description: '공기압이 비정상일 때 즉시 알림을 받으세요.',
      icon: Icons.notifications_active,
    ),
    OnboardingPage(
      title: '시작하기',
      description: '지금 바로 타이어 공기압 모니터링을 시작하세요.',
      icon: Icons.play_circle_filled,
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          PageView.builder(
            controller: _pageController,
            itemCount: _pages.length,
            onPageChanged: (int page) {
              setState(() {
                _currentPage = page;
              });
            },
            itemBuilder: (context, index) {
              return _buildPage(_pages[index]);
            },
          ),
          Positioned(
            bottom: 48,
            left: 0,
            right: 0,
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: List.generate(
                    _pages.length,
                    (index) => _buildDot(index),
                  ),
                ),
                const SizedBox(height: 32),
                if (_currentPage == _pages.length - 1)
                  ElevatedButton(
                    onPressed: () {
                      Navigator.pushReplacement(
                        context,
                        MaterialPageRoute(builder: (context) => FirstPage()),
                      );
                    },
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 48,
                        vertical: 16,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                    child: const Text(
                      '시작하기',
                      style: TextStyle(fontSize: 18),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPage(OnboardingPage page) {
    return Padding(
      padding: const EdgeInsets.all(40),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 200,
            height: 200,
            decoration: BoxDecoration(
              color: Colors.indigo.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
            child: Icon(
              page.icon,
              size: 100,
              color: Colors.indigo,
            ),
          ),
          const SizedBox(height: 40),
          Text(
            page.title,
            style: const TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
              color: Colors.indigo,
            ),
          ),
          const SizedBox(height: 20),
          Text(
            page.description,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 16,
              color: Colors.grey[600],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDot(int index) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 4),
      width: 8,
      height: 8,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: _currentPage == index ? Colors.indigo : Colors.grey[300],
      ),
    );
  }
}

class OnboardingPage {
  final String title;
  final String description;
  final IconData icon;

  OnboardingPage({
    required this.title,
    required this.description,
    required this.icon,
  });
}

class FirstPage extends StatefulWidget {
  FirstPage({Key? key}) : super(key: key);

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
        return Colors.red;
      case '공기압 높음':
        return Colors.orange;
      default:
        return Colors.green;
    }
  }

  IconData getStatusIcon(String status) {
    switch (status) {
      case '공기압 낮음':
        return Icons.arrow_downward;
      case '공기압 높음':
        return Icons.arrow_upward;
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
      appBar: AppBar(
        title: const Text(
          '타이어 공기압 모니터링',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
        leading: Builder(
          builder: (context) => IconButton(
            icon: const Icon(Icons.menu, color: Colors.white),
            onPressed: () => Scaffold.of(context).openDrawer(),
          ),
        ),
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            DrawerHeader(
              decoration: const BoxDecoration(color: Colors.indigo),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: const [
                  CircleAvatar(
                    radius: 30,
                    backgroundColor: Colors.white,
                    child: Icon(Icons.pedal_bike, size: 30, color: Colors.indigo),
                  ),
                  SizedBox(height: 10),
                  Text(
                    '타이어 공기압',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
            ListTile(
              leading: const Icon(Icons.notifications, color: Colors.indigo),
              title: const Text('알림설정'),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => AlarmPage()),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.settings, color: Colors.indigo),
              title: const Text('IP 설정'),
              onTap: () async {
                final result = await Navigator.push<double>(
                  context,
                  MaterialPageRoute(builder: (context) => FlaskPage()),
                );
                if (result != null) {
                  updatePressure(result);
                }
              },
            ),
          ],
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    children: [
                      const Icon(
                        Icons.pedal_bike,
                        size: 80,
                        color: Colors.indigo,
                      ),
                      const SizedBox(height: 20),
                      Text(
                        '${pressure.toStringAsFixed(1)} psi',
                        style: const TextStyle(
                          fontSize: 48,
                          fontWeight: FontWeight.bold,
                          color: Colors.indigo,
                        ),
                      ),
                      const SizedBox(height: 16),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        decoration: BoxDecoration(
                          color: statusColor.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(statusIcon, color: statusColor),
                            const SizedBox(width: 8),
                            Text(
                              status,
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: statusColor,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        '권장 공기압 범위',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          _buildPressureRange(
                            '최소',
                            minPressure,
                            Icons.arrow_downward,
                            Colors.orange,
                          ),
                          _buildPressureRange(
                            '최대',
                            maxPressure,
                            Icons.arrow_upward,
                            Colors.red,
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPressureRange(
      String label, double value, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          Icon(icon, color: color),
          const SizedBox(height: 8),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            '${value.toStringAsFixed(1)} psi',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}

class AlarmPage extends StatefulWidget {
  AlarmPage({Key? key}) : super(key: key);

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
        title: const Text('알람 설정', style: TextStyle(color: Colors.white)),
        backgroundColor: Colors.indigo,
        iconTheme: const IconThemeData(color: Colors.white),
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
                style: const TextStyle(fontSize: 16, color: Colors.indigo)),
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
                style: const TextStyle(fontSize: 16, color: Colors.indigo)),
            const SizedBox(height: 40),
            ListTile(
              title: const Text('푸시 알림 켜기/끄기', style: TextStyle(fontSize: 18)),
              trailing: Switch(
                value: _isPushEnabled,
                onChanged: (value) {
                  _togglePushSetting(value);
                },
                activeColor: Colors.indigo,
              ),
            ),
            const SizedBox(height: 40),
            ElevatedButton(
              onPressed: saveAlarmSettings,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.indigo,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
              ),
              child: const Text('저장', style: TextStyle(fontSize: 16)),
            ),
          ],
        ),
      ),
    );
  }
}

class FlaskPage extends StatefulWidget {
  FlaskPage({Key? key}) : super(key: key);

  @override
  _FlaskPageState createState() => _FlaskPageState();
}

class _FlaskPageState extends State<FlaskPage> {
  final TextEditingController _ipController = TextEditingController();
  String _responseMessage = '';
  bool _isLoading = false;

  Future<void> fetchData() async {
    final ip = _ipController.text.trim();
    if (ip.isEmpty) {
      setState(() {
        _responseMessage = 'IP를 입력하세요.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _responseMessage = '서버에 연결 중...';
    });

    final url = 'http://$ip:5000/api/data';
    try {
      final response = await http.get(Uri.parse(url)).timeout(
        const Duration(seconds: 10),
      );
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        // 서버에서 'pressure' 필드로 공기압 받아옴
        double pressure = (data['pressure'] as num).toDouble();

        setState(() {
          _responseMessage = '연결 성공!\n현재 공기압: ${pressure.toStringAsFixed(2)} psi';
          _isLoading = false;
        });

        // 첫 페이지로 공기압 값 전달하며 닫기
        Navigator.pop(context, pressure);
      } else {
        setState(() {
          _responseMessage = '서버 오류: ${response.statusCode}';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _responseMessage = '연결 실패: $e\n\n서버가 실행 중인지 확인하세요.';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('IP 설정 페이지', style: TextStyle(color: Colors.white)),
        backgroundColor: Colors.indigo,
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text('Flask 서버 IP를 입력하세요', 
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 12),
            TextField(
              controller: _ipController,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: '예: 192.168.0.101 또는 localhost',
                prefixIcon: Icon(Icons.computer, color: Colors.indigo),
              ),
              keyboardType: TextInputType.text,
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isLoading ? null : fetchData,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.indigo,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
              ),
              child: _isLoading 
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                  )
                : const Text('서버에 요청하기', style: TextStyle(fontSize: 16)),
            ),
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _responseMessage.isEmpty ? '서버 응답이 여기에 표시됩니다.' : _responseMessage,
                style: const TextStyle(fontSize: 16, color: Colors.black87),
                textAlign: TextAlign.center,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
