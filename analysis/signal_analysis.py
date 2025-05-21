import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os
from google.colab import files
import io

class SignalAnalyzer:
    def __init__(self):
        """
        신호 분석기 초기화
        - 샘플링 간격: 35ms
        - 샘플링 주파수: ~28.57 Hz (1/0.035)
        - 나이퀴스트 주파수: ~14.29 Hz (샘플링 주파수/2)
        """
        self.sampling_interval = 0.035  # 35ms
        self.sampling_freq = 1 / self.sampling_interval  # ~28.57 Hz
        self.nyquist_freq = self.sampling_freq / 2  # ~14.29 Hz
        
    def load_data(self, file_content):
        """
        업로드된 CSV 파일의 내용을 데이터프레임으로 로드

        Parameters:
        - file_content: Google Colab에서 업로드된 파일의 byte 내용

        Returns:
        - pandas DataFrame: 로드된 데이터
        """
        # CSV 파일을 데이터프레임으로 변환
        # file_content는 이미 byte 타입이므로 바로 io.BytesIO로 래핑
        df = pd.read_csv(io.BytesIO(file_content))
        return df
    
    def preprocess_signal(self, data):
        """
        신호 전처리 수행
        1. 디트렌딩: 선형 트렌드 제거
        2. 윈도잉: Hanning 윈도우 적용하여 스펙트럴 누설 감소
        
        Parameters:
        - data: 전처리할 신호 데이터 (numpy array)
        
        Returns:
        - windowed: 전처리된 신호 데이터
        """
        # 선형 트렌드 제거
        detrended = signal.detrend(data)
        
        # Hanning 윈도우 적용
        window = signal.windows.hann(len(detrended))
        windowed = detrended * window
        
        return windowed
    
    def compute_psd(self, data):
        """
        Welch 방법을 사용하여 파워 스펙트럼 밀도(PSD) 계산
        
        Parameters:
        - data: PSD를 계산할 신호 데이터
        
        Returns:
        - freqs: 주파수 배열
        - psd: 파워 스펙트럼 밀도 배열
        """
        # Welch 방법으로 PSD 계산
        freqs, psd = signal.welch(data, 
                                 fs=self.sampling_freq,
                                 nperseg=256,  # 세그먼트 길이
                                 noverlap=128,  # 세그먼트 간 중첩
                                 window='hann')  # Hanning 윈도우 사용
        return freqs, psd
    
    def plot_time_series(self, df, column):
        """
        시계열 데이터 시각화

        Parameters:
        - df: 데이터프레임
        - column: 시각화할 컬럼명
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df[column])
        plt.title(f'Time Series - {column}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
    
    def plot_psd(self, freqs, psd, column):
        """
        파워 스펙트럼 밀도(PSD) 시각화

        Parameters:
        - freqs: 주파수 배열
        - psd: 파워 스펙트럼 밀도 배열
        - column: 분석 중인 센서 컬럼명
        """
        plt.figure(figsize=(12, 6))
        plt.semilogy(freqs, psd)  # 로그 스케일로 PSD 표시
        plt.title(f'Power Spectral Density - {column}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density')
        plt.grid(True)
        # 나이퀴스트 주파수 표시
        plt.axvline(x=self.nyquist_freq, color='r', linestyle='--',
                   label=f'Nyquist Frequency ({self.nyquist_freq:.2f} Hz)')
        plt.legend()
        plt.show()
    
    def analyze_signal(self, df):
        """
        전체 신호 분석 수행

        Parameters:
        - df: 분석할 데이터프레임
        """
        # 분석할 센서 컬럼들
        columns = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']

        for column in columns:
            print(f"\nAnalyzing {column}...")

            # 시계열 분석
            self.plot_time_series(df, column)

            # 신호 전처리
            processed_data = self.preprocess_signal(df[column].values)

            # 주파수 분석
            freqs, psd = self.compute_psd(processed_data)

            # PSD 시각화
            self.plot_psd(freqs, psd, column)

            # 주파수 대역 정보 출력
            print(f"Sampling Frequency: {self.sampling_freq:.2f} Hz")
            print(f"Nyquist Frequency: {self.nyquist_freq:.2f} Hz")
            print(f"Frequency Resolution: {freqs[1]-freqs[0]:.3f} Hz")

def main():
    """
    메인 실행 함수
    - Google Colab에서 CSV 파일 업로드
    - 업로드된 파일 분석
    """
    # 신호 분석기 초기화
    analyzer = SignalAnalyzer()
    
    # 파일 업로드 요청
    print("분석할 CSV 파일을 업로드하세요...")
    uploaded = files.upload()
    
    if not uploaded:
        print("파일이 업로드되지 않았습니다.")
        return
    
    # 업로드된 파일 처리
    for filename, file_content in uploaded.items():
        print(f"\n{filename} 파일 분석 중...")
        df = analyzer.load_data(file_content)
        analyzer.analyze_signal(df)

if __name__ == "__main__":
    main() 