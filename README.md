# TIRE (Tire Inspection and Real-time Evaluation)

자전거 타이어 상태를 실시간으로 모니터링하고 분석하는 시스템입니다. 센서 데이터를 기반으로 타이어의 마모도와 상태를 분석하여, 사용자가 모바일 앱을 통해 직관적으로 확인할 수 있도록 합니다.

## 프로젝트 배경

자전거는 일상적인 교통수단이자 여가 활동의 수단으로 널리 사용되고 있지만, 타이어 상태는 대부분 사용자 경험에 의존하여 점검되고 있습니다. 이로 인해 마모나 공기압 저하 같은 요소를 제때 인지하지 못하는 경우가 종종 발생하며, 이는 주행 효율이나 안전에 영향을 줄 수 있습니다.

본 프로젝트는 센서 데이터를 기반으로 자전거 타이어의 상태를 분석하고, 이를 모바일 앱을 통해 직관적으로 확인할 수 있는 시스템을 개발하는 것을 목표로 합니다. 가속도 센서와 마이크로컨트롤러를 활용해 주행 데이터를 수집하고, 머신러닝 모델을 통해 타이어 마모도를 추정하며, 분석 결과를 앱에서 실시간으로 제공합니다.

## 주요 기능

- 실시간 타이어 상태 모니터링
- 센서 데이터 기반 마모도도 분석
- 머신러닝을 통한 타이어 상태 예측
- 모바일 앱을 통한 직관적인 상태 확인

## 시스템 구성
- **하드웨어**
  - MPU6050 센서 (가속도/자이로)
  - ESP32-C3 마이크로컨트롤러 (송신/수신)
  - 라즈베리파이4 (데이터 처리)

- **소프트웨어**
  - Arduino IDE (ESP32-C3 프로그래밍)
  - Python (데이터 처리 및 AI 모델)
  - Flutter (모바일 앱)

## 프로젝트 구조
```
TIRE/
├── esp32/
│   ├── sender/          # ESP32-C3 송신 코드
│   └── broker/        # ESP32-C3 수신 코드
├── raspberry_pi/
│   ├── data_collector/  # UART 데이터 수신 및 CSV 저장
│   └── server/       # 웹소켓 서버
├── analysis/            # 데이터 분석 코드
└── flutter_app/         # Flutter 모바일 앱
```

## 설치 및 설정

### 1. 하드웨어 설정
1. MPU6050 센서와 ESP32-C3 연결
   - VCC → 3.3V
   - GND → GND
   - SCL → GPIO05
   - SDA → GPIO04

2. ESP32-C3와 라즈베리파이 연결
   - ESP32-C3 TX → 라즈베리파이 RX
   - ESP32-C3 RX → 라즈베리파이 TX
   - GND → GND

3. 라즈베리파이 데이터 수집 버튼 및 확인용 LED
   - button → GPIO17
   - LED → GPIO18

### 2. 소프트웨어 설정

#### ESP32-C3 (Arduino IDE)
1. 필요한 라이브러리 설치, 보드 추가
   - Adafruit MPU6050
   - esp32 by Espressif Systems

2. 코드 업로드
   - `esp32/esp32_sender.ino` → 송신용 ESP32-C3
   - `esp32/esp32_broker.ino` → 수신용 ESP32-C3

#### 라즈베리파이
1. Python 패키지 설치
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn tensorflow joblib websockets
```

2. 웹소켓 서버 실행
```bash
cd raspberry_pi/server.py
```

#### 모바일 앱 (Flutter)
1. Flutter 개발 환경 설정
```bash
flutter pub get
```

2. 앱 실행
```bash
flutter run
```

## 사용 방법

### 1. 시스템 시작
1. ESP32-C3 전원 켜기
2. 라즈베리파이에서 웹소켓 서버 실행
3. 모바일 앱 실행

### 2. 모니터링
- 앱에서 실시간 타이어 상태 확인
- 상태 변화 시 알림 수신
- 주행 이력 확인

## 확장 가능성

비록 실험은 개인 자전거를 중심으로 이루어지지만, 이러한 접근 방식은 공공 자전거 관리나 공유 모빌리티 안전성 향상 등 다양한 분야로 확장 가능성이 있습니다. 본 연구는 소규모 환경에서 출발했지만, 센서 기반 상태 진단 기술이 일상 속에서 어떻게 활용될 수 있는지를 실험적으로 탐색하는 데 의의가 있습니다.

## 참고 자료
- [MPU6050 데이터시트](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Datasheet1.pdf)
- [ESP32-C3 기술문서](https://www.espressif.com/en/products/socs/esp32-c3)
- [TensorFlow 문서](https://www.tensorflow.org/)
- [Flutter 문서](https://flutter.dev/docs) 
