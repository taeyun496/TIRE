// ------------------------------
//  설정 변수 (상단 관리)
// ------------------------------
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <esp_now.h>
#include <WiFi.h>

// I2C 핀
#define SDA_PIN 4
#define SCL_PIN 5

// 수집 주기
#define SAMPLE_INTERVAL_MS 7

// 수신기 MAC 주소
uint8_t receiverMac[] = {0x9C, 0x9E, 0x6E, 0xB8, 0x8F, 0xE4};

// ------------------------------
//  구조체 정의
// ------------------------------
typedef struct {
  unsigned long timestamp;
  float accelX, accelY, accelZ;
  float gyroX, gyroY, gyroZ;
} SensorPacket;

SensorPacket dataPacket;

// ------------------------------
//  센서 및 통신 객체
// ------------------------------
Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);

  // I2C 시작
  Wire.begin(SDA_PIN, SCL_PIN);

  // MPU6050 초기화
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("MPU6050 not found!");
    while (1) delay(10);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // ESP-NOW 초기화
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed!");
    while (1);
  }

  // 피어 등록
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    while (1);
  }

  Serial.println(" Sender Ready.");
}

void loop() {
  // 센서 읽기
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  dataPacket.timestamp = millis();
  dataPacket.accelX = a.acceleration.x;
  dataPacket.accelY = a.acceleration.y;
  dataPacket.accelZ = a.acceleration.z;
  dataPacket.gyroX = g.gyro.x;
  dataPacket.gyroY = g.gyro.y;
  dataPacket.gyroZ = g.gyro.z;

  // ESP-NOW 전송
  esp_err_t result = esp_now_send(receiverMac, (uint8_t *)&dataPacket, sizeof(dataPacket));
  if (result == ESP_OK) {
    Serial.println(" Data sent");
  } else {
    Serial.println(" Send failed");
  }

  delay(SAMPLE_INTERVAL_MS);  // 데이터 수집 주기 조절
}
