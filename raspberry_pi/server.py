import asyncio
import websockets
import json
from data_analyzer import DataAnalyzer
import serial
import numpy as np
import pandas as pd

class WebSocketServer:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.analyzer = DataAnalyzer()
        self.analyzer.load_model()  # 학습된 모델 로드
        self.serial = serial.Serial('/dev/ttyS0', 115200, timeout=1)
        self.buffer = []
        self.window_size = 100
        
    async def process_data(self, websocket):
        """실시간 데이터 처리 및 예측"""
        while True:
            if self.serial.in_waiting:
                line = self.serial.readline().decode('utf-8').strip()
                if line:
                    try:
                        # 데이터 파싱
                        data = [float(x) for x in line.split(',')]
                        self.buffer.append(data[1:])  # timestamp 제외
                        
                        # 윈도우 크기만큼 데이터가 모이면 예측
                        if len(self.buffer) >= self.window_size:
                            features = self.analyzer.extract_features(
                                pd.DataFrame(self.buffer, 
                                          columns=['accelX', 'accelY', 'accelZ',
                                                 'gyroX', 'gyroY', 'gyroZ'])
                            )
                            
                            prediction = self.analyzer.predict(features[-1:])
                            status = "normal" if prediction[0][0] < 0.5 else "worn"
                            
                            # 결과 전송
                            await websocket.send(json.dumps({
                                'status': status,
                                'confidence': float(prediction[0][0]),
                                'timestamp': data[0]
                            }))
                            
                            # 버퍼 업데이트
                            self.buffer = self.buffer[-self.window_size:]
                            
                    except ValueError:
                        print(f"Invalid data format: {line}")
                        
            await asyncio.sleep(0.1)  # CPU 사용량 조절
            
    async def handler(self, websocket, path):
        """웹소켓 연결 핸들러"""
        try:
            await self.process_data(websocket)
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
            
    def run(self):
        """서버 실행"""
        start_server = websockets.serve(self.handler, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    server = WebSocketServer()
    print(f"Starting WebSocket server on ws://{server.host}:{server.port}")
    server.run() 