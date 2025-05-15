import serial
import csv
import os
import time
from datetime import datetime
import RPi.GPIO as GPIO
import threading

# ------------------------------
#  Setup variables (easy to change)
# ------------------------------
SERIAL_PORT = '/dev/serial0'  # UART port name
BAUD_RATE = 115200            # UART speed
BUTTON_PIN = 17               # Button pin (GPIO17)
LED_PIN = 18                  # LED pin (GPIO18)
DATA_DIR = 'sensor_data'      # Folder to save files
COLLECTION_TIME = 60          # Save time in seconds
DATA_INTERVAL = 0.035         # Expected gap between data lines (seconds)
EXPECTED_COLUMNS = 8          # Total data parts (1 prefix + 7 values)
DEBUG_MODE = True             # Show print log or not

# ------------------------------
#  Data logger class
# ------------------------------
class UARTLogger:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.output(LED_PIN, GPIO.LOW)

        # Start UART
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)

        # Make folder if not exist
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        self.collecting = False
        self.file = None

    def generate_filename(self):
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(DATA_DIR, f"mpu_data_{now}.csv")

    def start_collection(self):
        self.collecting = True
        start_time = time.time()
        file_path = self.generate_filename()
        GPIO.output(LED_PIN, GPIO.HIGH)
        data_count = 0
        self.ser.reset_input_buffer()

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'accelX', 'accelY', 'accelZ',
                             'gyroX', 'gyroY', 'gyroZ'])

            while time.time() - start_time < COLLECTION_TIME:
                if self.ser.in_waiting:
                    try:
                        # Read one line
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        if DEBUG_MODE and line:
                            print(f" {line}")

                        if line.startswith("M"):
                            fields = line[2:].split(",")  # skip "M,"
                            if len(fields) == 7:
                                writer.writerow(fields)
                                data_count += 1
                            else:
                                if DEBUG_MODE:
                                    print(f" Wrong column count: {fields}")
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f" Read error: {e}")

        GPIO.output(LED_PIN, GPIO.LOW)
        self.collecting = False

        # Count how much data was saved
        expected = int(COLLECTION_TIME / DATA_INTERVAL)
        completeness = (data_count / expected) * 100
        print(f"\n File saved: {file_path}")
        print(f" Data lines: {data_count}, Expected: {expected}")
        print(f" Save rate: {completeness:.2f}%")

    def button_callback(self, channel):
        if not self.collecting:
            threading.Thread(target=self.start_collection).start()

    def run(self):
        GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=self.button_callback, bouncetime=300)
        print(" Press button to start saving data...")

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n Program stopped")
        finally:
            if self.ser.is_open:
                self.ser.close()
            GPIO.cleanup()

# ------------------------------
#  Run the logger
# ------------------------------
if __name__ == "__main__":
    logger = UARTLogger()
    logger.run()