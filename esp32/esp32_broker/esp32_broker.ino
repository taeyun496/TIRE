// ------------------------------
//  설정 변수 (상단 관리)
// ------------------------------
#include <esp_now.h>
#include <WiFi.h>
#include <HardwareSerial.h>

// UART 핀
#define UART_RX_PIN 20
#define UART_TX_PIN 21
#define UART_BAUD 115200

// 구조체 정의 (송신기와 반드시 동일)
typedef struct {
  unsigned long timestamp;
  float accelX, accelY, accelZ;
  float gyroX, gyroY, gyroZ;
} SensorPacket;

SensorPacket receivedData;
HardwareSerial SerialPi(1);  // UART1 객체 생성

// ------------------------------
//  ESP-NOW 수신 콜백
// ------------------------------
void onDataReceive(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  if (len == sizeof(SensorPacket)) {
    memcpy(&receivedData, incomingData, sizeof(receivedData));

    // USB 시리얼로 디버깅 출력
    Serial.printf(" %lu | A: %.2f %.2f %.2f | G: %.2f %.2f %.2f\n",
                  receivedData.timestamp,
                  receivedData.accelX, receivedData.accelY, receivedData.accelZ,
                  receivedData.gyroX, receivedData.gyroY, receivedData.gyroZ);

    // 라즈베리파이로 UART 전송 (CSV 포맷)
    char line[128];
    snprintf(line, sizeof(line), "M,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
             receivedData.timestamp,
             receivedData.accelX, receivedData.accelY, receivedData.accelZ,
             receivedData.gyroX, receivedData.gyroY, receivedData.gyroZ);
    SerialPi.print(line);
  } else {
    Serial.printf(" Invalid data size: %d bytes\n", len);
  }
}

void setup() {
  Serial.begin(115200);  // USB 디버깅
  SerialPi.begin(UART_BAUD, SERIAL_8N1, UART_RX_PIN, UART_TX_PIN);  // UART1

  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println(" ESP-NOW init failed!");
    while (true);
  }

  esp_now_register_recv_cb(onDataReceive);
  Serial.println(" Receiver Ready");
}

void loop() {
  delay(100);  // 이벤트 기반, 루프는 비워도 됨
}