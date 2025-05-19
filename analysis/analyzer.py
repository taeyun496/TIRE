import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# 데이터 로드
def load_data(file_path):
    """CSV 파일에서 센서 데이터 로드"""
    df = pd.read_csv(file_path)
    print(f"데이터 형태: {df.shape}")
    print(f"컬럼명: {df.columns.tolist()}")
    return df

# 데이터 전처리
def preprocess_data(df):
    """데이터 전처리: 타임스탬프 처리 및 기본 통계 계산"""
    # 타임스탬프 컬럼을 datetime으로 변환 (첫 번째 컬럼이 타임스탬프라고 가정)
    timestamp_col = df.columns[0]
    
    # 타임스탬프가 문자열 형식이면 datetime으로 변환
    if df[timestamp_col].dtype == 'object':
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except:
            print("타임스탬프 변환 실패. 원래 형식 유지.")
    
    # 센서 데이터 컬럼 (첫 번째 컬럼 제외)
    sensor_columns = df.columns[1:]
    
    # 기본 통계량 계산
    stats = df[sensor_columns].describe()
    print("\n기본 통계량:")
    print(stats)
    
    return df, sensor_columns

# FFT 분석 함수
def perform_fft_analysis(df, sensor_columns, sampling_period_ms=35):
    """
    각 센서 데이터에 대한 FFT 분석 수행
    
    Parameters:
        df (DataFrame): 센서 데이터가 포함된 DataFrame
        sensor_columns (list): 분석할 센서 데이터 컬럼명 리스트
        sampling_period_ms (float): 샘플링 주기 (밀리초)
    
    Returns:
        dict: 각 센서별 FFT 결과가 저장된 딕셔너리
    """
    # 샘플링 주파수 계산 (Hz)
    sampling_freq = 1000 / sampling_period_ms  # ms -> Hz 변환
    print(f"\n샘플링 주파수: {sampling_freq} Hz")
    
    # 데이터 길이
    N = len(df)
    print(f"데이터 포인트 수: {N}")
    
    # 분석 결과 저장용 딕셔너리
    fft_results = {}
    
    for column in sensor_columns:
        # 센서 데이터 추출
        sensor_data = df[column].values
        
        # 데이터에서 선형 추세 제거 (디트렌딩)
        detrended_data = signal.detrend(sensor_data)
        
        # 해밍 윈도우 적용 (스펙트럼 누설 감소)
        windowed_data = detrended_data * np.hamming(N)
        
        # FFT 계산
        fft_data = fft(windowed_data)
        
        # 주파수 배열 계산
        freqs = fftfreq(N, d=sampling_period_ms/1000)
        
        # 양의 주파수 부분만 선택 (nyquist 주파수까지)
        positive_freq_idxs = np.where(freqs > 0)
        freqs_pos = freqs[positive_freq_idxs]
        fft_pos = np.abs(fft_data)[positive_freq_idxs]
        
        # 정규화된 파워 스펙트럼 밀도 계산
        fft_pos_normalized = fft_pos / N
        
        # 주파수와 해당 파워를 결과 딕셔너리에 저장
        fft_results[column] = {
            'frequencies': freqs_pos,
            'amplitudes': fft_pos_normalized,
            'raw_data': sensor_data,
            'detrended_data': detrended_data
        }
        
        # 주요 주파수 성분 식별
        dominant_idx = np.argmax(fft_pos_normalized)
        dominant_freq = freqs_pos[dominant_idx]
        dominant_amp = fft_pos_normalized[dominant_idx]
        
        print(f"\n{column} 센서의 주요 주파수 성분: {dominant_freq:.2f} Hz (진폭: {dominant_amp:.6f})")
        
    return fft_results

# 시각화 함수
def visualize_fft_results(fft_results, sensor_columns):
    """FFT 분석 결과 시각화"""
    # 서브플롯 설정
    n_columns = len(sensor_columns)
    fig, axes = plt.subplots(n_columns, 2, figsize=(15, 4*n_columns))
    
    for i, column in enumerate(sensor_columns):
        result = fft_results[column]
        
        # 타임 도메인 시각화
        if n_columns > 1:
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]
        else:
            ax1 = axes[0]
            ax2 = axes[1]
            
        ax1.plot(result['raw_data'], 'b-', alpha=0.5, label='원본 데이터')
        ax1.plot(result['detrended_data'], 'r-', label='전처리된 데이터')
        ax1.set_title(f'{column} - 시간 도메인')
        ax1.set_xlabel('샘플 인덱스')
        ax1.set_ylabel('진폭')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 주파수 도메인 시각화 (로그 스케일)
        ax2.plot(result['frequencies'], result['amplitudes'])
        ax2.set_title(f'{column} - 주파수 도메인')
        ax2.set_xlabel('주파수 (Hz)')
        ax2.set_ylabel('정규화된 진폭')
        ax2.grid(True, alpha=0.3)
        
        # x축을 최대 주파수의 절반까지만 표시 (가독성 향상)
        max_freq = result['frequencies'].max()
        ax2.set_xlim(0, max_freq/2)
        
        # y축 로그 스케일 적용 (작은 피크 식별 용이)
        ax2.set_yscale('log')
        
        # 주요 주파수 위치 표시
        dominant_idx = np.argmax(result['amplitudes'])
        dominant_freq = result['frequencies'][dominant_idx]
        dominant_amp = result['amplitudes'][dominant_idx]
        
        ax2.axvline(x=dominant_freq, color='r', linestyle='--', alpha=0.7)
        ax2.text(dominant_freq*1.1, dominant_amp, f'{dominant_freq:.2f} Hz', 
                 color='r', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('fft_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 추가 분석: 파워 스펙트럼 밀도 추정
def calculate_psd(df, sensor_columns, sampling_period_ms=35):
    """Welch 방법을 사용한 파워 스펙트럼 밀도 추정"""
    sampling_freq = 1000 / sampling_period_ms
    
    psd_results = {}
    
    for column in sensor_columns:
        sensor_data = df[column].values
        
        # Welch 방법으로 PSD 추정
        frequencies, Pxx = signal.welch(sensor_data, fs=sampling_freq, 
                                        nperseg=1024, scaling='density')
        
        psd_results[column] = {
            'frequencies': frequencies,
            'psd': Pxx
        }
    
    return psd_results

# 주파수 대역별 에너지 분포 계산
def analyze_frequency_bands(fft_results, sensor_columns):
    """주파수 대역별 에너지 분포 분석"""
    bands = {
        '0-1 Hz': (0, 1),
        '1-5 Hz': (1, 5),
        '5-10 Hz': (5, 10),
        '10-20 Hz': (10, 20),
        '>20 Hz': (20, float('inf'))
    }
    
    band_energies = {column: {} for column in sensor_columns}
    
    for column in sensor_columns:
        freqs = fft_results[column]['frequencies']
        amps = fft_results[column]['amplitudes']
        
        total_energy = np.sum(amps**2)
        
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs < high)
            if np.any(band_mask):
                band_energy = np.sum(amps[band_mask]**2)
                percentage = (band_energy / total_energy) * 100
                band_energies[column][band_name] = percentage
            else:
                band_energies[column][band_name] = 0
    
    return band_energies

# 주요 분석 함수
def main(file_path, sampling_period_ms=35):
    """메인 분석 함수"""
    # 데이터 로드
    df = load_data(file_path)
    
    # 데이터 전처리
    df, sensor_columns = preprocess_data(df)
    
    # FFT 분석 수행
    fft_results = perform_fft_analysis(df, sensor_columns, sampling_period_ms)
    
    # 시각화
    visualize_fft_results(fft_results, sensor_columns)
    
    # 파워 스펙트럼 밀도 추정
    psd_results = calculate_psd(df, sensor_columns, sampling_period_ms)
    
    # 주파수 대역별 에너지 분포 분석
    band_energies = analyze_frequency_bands(fft_results, sensor_columns)
    
    # 주파수 대역 에너지 시각화
    plt.figure(figsize=(12, 8))
    
    band_names = list(next(iter(band_energies.values())).keys())
    x = np.arange(len(band_names))
    width = 0.8 / len(sensor_columns)
    
    for i, column in enumerate(sensor_columns):
        energies = [band_energies[column][band] for band in band_names]
        plt.bar(x + i*width, energies, width, label=column)
    
    plt.xlabel('주파수 대역')
    plt.ylabel('에너지 비율 (%)')
    plt.title('센서별 주파수 대역 에너지 분포')
    plt.xticks(x + width * (len(sensor_columns) - 1) / 2, band_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('frequency_band_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fft_results, psd_results, band_energies

if __name__ == "__main__":
    # CSV 파일 경로 지정
    file_path = "sensor_data.csv"
    
    # 샘플링 주기 35ms 설정
    sampling_period_ms = 35
    
    # 분석 실행
    fft_results, psd_results, band_energies = main(file_path, sampling_period_ms) 