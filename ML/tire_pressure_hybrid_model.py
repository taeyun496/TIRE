# Google Colab용 타이어 공기압 예측 CNN+LSTM 하이브리드 모델
# 사용법: Google Colab에서 이 코드를 복사하여 실행하세요

# 필요한 패키지 설치
# !pip install tensorflow numpy pandas scikit-learn matplotlib seaborn scipy

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Concatenate, BatchNormalization, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis
import os
import glob
import warnings
import zipfile
from google.colab import files
import io

warnings.filterwarnings('ignore')

# GPU 사용 가능 여부 확인
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

class TirePressureHybridPredictor:
    def __init__(self, sequence_length=1000, step_size=100):
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.scaler_raw = StandardScaler()
        self.scaler_features = StandardScaler()
        self.model = None
        # 예측 범위 제한 (32-64 psi)
        self.min_pressure = 32
        self.max_pressure = 64
        
    def upload_and_extract_data(self):
        """Google Colab에서 데이터 파일 업로드 및 압축 해제"""
        print("데이터 파일을 업로드해주세요.")
        print("옵션 1: mpudata 폴더를 zip으로 압축하여 업로드")
        print("옵션 2: CSV 파일들을 개별적으로 업로드")
        
        choice = input("압축 파일을 업로드하시겠습니까? (y/n): ").lower()
        
        if choice == 'y':
            print("ZIP 파일을 선택해주세요:")
            uploaded = files.upload()
            
            # ZIP 파일 압축 해제
            for filename in uploaded.keys():
                if filename.endswith('.zip'):
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall('.')
                    print(f"압축 해제 완료: {filename}")
                    
                    # mpudata 폴더 확인
                    if os.path.exists('mpudata'):
                        print("mpudata 폴더를 찾았습니다.")
                        return 'mpudata'
                    else:
                        # 압축 해제된 파일들 확인
                        csv_files = glob.glob('*.csv')
                        if csv_files:
                            os.makedirs('mpudata', exist_ok=True)
                            for csv_file in csv_files:
                                if 'mpu_data' in csv_file and 'psi' in csv_file:
                                    os.rename(csv_file, f'mpudata/{csv_file}')
                            print("CSV 파일들을 mpudata 폴더로 이동했습니다.")
                            return 'mpudata'
        else:
            print("CSV 파일들을 선택해주세요 (여러 파일 선택 가능):")
            uploaded = files.upload()
            
            # 업로드된 파일들을 mpudata 폴더로 이동
            os.makedirs('mpudata', exist_ok=True)
            for filename in uploaded.keys():
                if filename.endswith('.csv') and 'mpu_data' in filename and 'psi' in filename:
                    # 파일을 mpudata 폴더로 이동
                    with open(f'mpudata/{filename}', 'wb') as f:
                        f.write(uploaded[filename])
                    print(f"파일 저장: mpudata/{filename}")
            
            return 'mpudata'
        
        return None
    
    def load_data(self, data_folder='mpudata'):
        """데이터 로드 및 전처리"""
        all_data = []
        all_labels = []
        
        # 폴더가 없으면 업로드 시도
        if not os.path.exists(data_folder):
            print(f"{data_folder} 폴더를 찾을 수 없습니다. 데이터를 업로드해주세요.")
            data_folder = self.upload_and_extract_data()
            if data_folder is None:
                raise ValueError("데이터 폴더를 찾을 수 없습니다.")
        
        # 각 압력별 파일 로드
        for pressure in [35, 50, 60]:
            files_list = glob.glob(f"{data_folder}/mpu_data_{pressure}psi_*.csv")
            print(f"Loading {pressure}psi data: {len(files_list)} files")
            
            if len(files_list) == 0:
                print(f"Warning: {pressure}psi 데이터 파일을 찾을 수 없습니다.")
                continue
            
            for file in files_list:
                try:
                    df = pd.read_csv(file)
                    print(f"Loaded {file}: {len(df)} rows")
                    
                    # 센서 데이터만 추출 (timestamp 제외)
                    sensor_data = df[['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']].values
                    
                    # 슬라이딩 윈도우로 시퀀스 생성
                    sequences_from_file = 0
                    for i in range(0, len(sensor_data) - self.sequence_length + 1, self.step_size):
                        sequence = sensor_data[i:i + self.sequence_length]
                        all_data.append(sequence)
                        all_labels.append(pressure)
                        sequences_from_file += 1
                    
                    print(f"  Generated {sequences_from_file} sequences from {file}")
                    
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        if len(all_data) == 0:
            raise ValueError("데이터를 로드할 수 없습니다. 파일 형식을 확인해주세요.")
        
        return np.array(all_data), np.array(all_labels)
    
    def extract_time_domain_features(self, sequence):
        """시간 도메인 특성 추출"""
        features = []
        
        for channel in range(sequence.shape[1]):  # 6개 채널
            data = sequence[:, channel]
            
            # 기본 통계량
            features.extend([
                np.mean(data),           # 평균
                np.std(data),            # 표준편차
                np.max(data),            # 최대값
                np.min(data),            # 최소값
                np.sqrt(np.mean(data**2)), # RMS
                skew(data),              # 왜도
                kurtosis(data),          # 첨도
                np.ptp(data),            # Peak-to-peak
                np.percentile(data, 25), # 1사분위수
                np.percentile(data, 75), # 3사분위수
            ])
        
        return np.array(features)
    
    def extract_frequency_domain_features(self, sequence, fs=28.6):
        """주파수 도메인 특성 추출"""
        features = []
        
        for channel in range(sequence.shape[1]):  # 6개 채널
            data = sequence[:, channel]
            
            # FFT 계산
            fft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data), 1/fs)
            
            # 양의 주파수만 사용
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # 주파수 대역별 파워
            freq_bands = [(0, 2), (2, 5), (5, 10), (10, 15)]
            for low, high in freq_bands:
                band_mask = (positive_freqs >= low) & (positive_freqs < high)
                band_power = np.sum(positive_fft[band_mask]**2)
                features.append(band_power)
            
            # 지배 주파수
            dominant_freq_idx = np.argmax(positive_fft)
            dominant_freq = positive_freqs[dominant_freq_idx]
            features.append(dominant_freq)
            
            # 스펙트럼 중심 주파수
            spectral_centroid = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
            features.append(spectral_centroid)
            
            # 총 파워
            total_power = np.sum(positive_fft**2)
            features.append(total_power)
            
            # 스펙트럼 롤오프 (95% 에너지 포함하는 주파수)
            cumsum_power = np.cumsum(positive_fft**2)
            rolloff_idx = np.where(cumsum_power >= 0.95 * total_power)[0]
            if len(rolloff_idx) > 0:
                spectral_rolloff = positive_freqs[rolloff_idx[0]]
            else:
                spectral_rolloff = positive_freqs[-1]
            features.append(spectral_rolloff)
        
        return np.array(features)
    
    def extract_all_features(self, sequences):
        """모든 특성 추출"""
        time_features = []
        freq_features = []
        
        print("Extracting features...")
        for i, seq in enumerate(sequences):
            if i % 100 == 0:
                print(f"Processing sequence {i}/{len(sequences)}")
            
            time_feat = self.extract_time_domain_features(seq)
            freq_feat = self.extract_frequency_domain_features(seq)
            
            time_features.append(time_feat)
            freq_features.append(freq_feat)
        
        return np.array(time_features), np.array(freq_features)
    
    def build_model(self, input_shape_raw, input_shape_features):
        """CNN+LSTM 하이브리드 모델 구축 (32-64 psi 범위로 제한)"""
        # 원시 데이터 입력
        raw_input = Input(shape=input_shape_raw, name='raw_input')
        
        # CNN 브랜치 - 공간적 특성 추출
        cnn_branch = Conv1D(32, 3, activation='relu', padding='same')(raw_input)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(2)(cnn_branch)
        
        cnn_branch = Conv1D(64, 3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(2)(cnn_branch)
        
        cnn_branch = Conv1D(128, 3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(2)(cnn_branch)
        
        # CNN 출력을 LSTM에 전달
        # LSTM 브랜치 - 시간적 의존성 학습
        lstm_branch = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(cnn_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
        
        lstm_branch = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2))(lstm_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
        
        # 추가 CNN 브랜치 - 글로벌 특성 추출
        global_cnn = Conv1D(64, 5, activation='relu', padding='same')(raw_input)
        global_cnn = BatchNormalization()(global_cnn)
        global_cnn = MaxPooling1D(4)(global_cnn)
        
        global_cnn = Conv1D(128, 5, activation='relu', padding='same')(global_cnn)
        global_cnn = BatchNormalization()(global_cnn)
        global_cnn = GlobalAveragePooling1D()(global_cnn)
        
        # 특성 입력 브랜치
        feature_input = Input(shape=input_shape_features, name='feature_input')
        feature_branch = Dense(128, activation='relu')(feature_input)
        feature_branch = BatchNormalization()(feature_branch)
        feature_branch = Dropout(0.3)(feature_branch)
        
        feature_branch = Dense(64, activation='relu')(feature_branch)
        feature_branch = BatchNormalization()(feature_branch)
        feature_branch = Dropout(0.3)(feature_branch)
        
        # 모든 브랜치 결합
        combined = Concatenate()([lstm_branch, global_cnn, feature_branch])
        
        # 최종 분류 레이어
        x = Dense(256, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # 회귀 출력 (32-64 psi 범위로 제한)
        output = Dense(1, activation='sigmoid', name='pressure_output')(x)
        # sigmoid 출력을 32-64 범위로 스케일링
        output = tf.keras.layers.Lambda(lambda x: x * (self.max_pressure - self.min_pressure) + self.min_pressure)(output)
        
        model = Model(inputs=[raw_input, feature_input], outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_raw, X_features, y, validation_split=0.1, epochs=100, batch_size=32):
        """모델 훈련 (더 많은 데이터를 훈련에 사용)"""
        # 데이터 분할
        X_raw_train, X_raw_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
            X_raw, X_features, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training data: {len(X_raw_train)} samples")
        print(f"Validation data: {len(X_raw_val)} samples")
        print(f"Training ratio: {len(X_raw_train)/(len(X_raw_train)+len(X_raw_val)):.1%}")
        
        # 데이터 정규화
        X_raw_train_scaled = self.scaler_raw.fit_transform(
            X_raw_train.reshape(-1, X_raw_train.shape[-1])
        ).reshape(X_raw_train.shape)
        
        X_raw_val_scaled = self.scaler_raw.transform(
            X_raw_val.reshape(-1, X_raw_val.shape[-1])
        ).reshape(X_raw_val.shape)
        
        X_feat_train_scaled = self.scaler_features.fit_transform(X_feat_train)
        X_feat_val_scaled = self.scaler_features.transform(X_feat_val)
        
        # 모델 구축
        self.model = self.build_model(
            input_shape_raw=(X_raw_train.shape[1], X_raw_train.shape[2]),
            input_shape_features=(X_feat_train.shape[1],)
        )
        
        print(self.model.summary())
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),  # 하이브리드는 더 오래 훈련
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6)
        ]
        
        # 훈련
        history = self.model.fit(
            [X_raw_train_scaled, X_feat_train_scaled], y_train,
            validation_data=([X_raw_val_scaled, X_feat_val_scaled], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 훈련 및 검증 데이터 저장 (나중에 테스트용)
        self.train_data = (X_raw_train_scaled, X_feat_train_scaled, y_train)
        self.val_data = (X_raw_val_scaled, X_feat_val_scaled, y_val)
        
        return history
    
    def predict(self, X_raw, X_features):
        """예측 (32-64 psi 범위로 제한)"""
        X_raw_scaled = self.scaler_raw.transform(
            X_raw.reshape(-1, X_raw.shape[-1])
        ).reshape(X_raw.shape)
        
        X_feat_scaled = self.scaler_features.transform(X_features)
        
        predictions = self.model.predict([X_raw_scaled, X_feat_scaled])
        
        # 예측값을 32-64 범위로 클리핑
        predictions = np.clip(predictions.flatten(), self.min_pressure, self.max_pressure)
        
        return predictions
    
    def evaluate_model(self, X_raw, X_features, y_true, dataset_name="Dataset"):
        """모델 평가"""
        y_pred = self.predict(X_raw, X_features)
        
        # 회귀 메트릭
        mae = mean_absolute_error(y_true, y_pred)
        mse = np.mean((y_true - y_pred)**2)
        rmse = np.sqrt(mse)
        
        # 분류 정확도 (±3 오차 허용)
        tolerance = 3
        correct_predictions = np.abs(y_true - y_pred) <= tolerance
        accuracy = np.mean(correct_predictions)
        
        print(f"\n=== {dataset_name} 평가 결과 (CNN+LSTM Hybrid) ===")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Square Error: {rmse:.2f}")
        print(f"Accuracy (±{tolerance} tolerance): {accuracy:.3f}")
        print(f"Prediction range: {y_pred.min():.1f} - {y_pred.max():.1f} psi")
        
        # 각 클래스별 정확도
        for pressure in [35, 50, 60]:
            mask = y_true == pressure
            if np.sum(mask) > 0:
                class_mae = np.mean(np.abs(y_pred[mask] - y_true[mask]))
                class_accuracy = np.mean(np.abs(y_pred[mask] - y_true[mask]) <= tolerance)
                print(f"{pressure}psi - MAE: {class_mae:.2f}, Accuracy: {class_accuracy:.3f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    def evaluate_on_train_val(self):
        """훈련 및 검증 데이터에 대한 평가"""
        if not hasattr(self, 'train_data') or not hasattr(self, 'val_data'):
            print("훈련 데이터가 없습니다. 먼저 모델을 훈련시켜주세요.")
            return None, None
        
        # 훈련 데이터 평가
        X_raw_train, X_feat_train, y_train = self.train_data
        train_results = self.evaluate_model_direct(X_raw_train, X_feat_train, y_train, "Training")
        
        # 검증 데이터 평가
        X_raw_val, X_feat_val, y_val = self.val_data
        val_results = self.evaluate_model_direct(X_raw_val, X_feat_val, y_val, "Validation")
        
        return train_results, val_results
    
    def evaluate_model_direct(self, X_raw_scaled, X_feat_scaled, y_true, dataset_name="Dataset"):
        """이미 스케일링된 데이터에 대한 직접 평가"""
        predictions = self.model.predict([X_raw_scaled, X_feat_scaled])
        y_pred = np.clip(predictions.flatten(), self.min_pressure, self.max_pressure)
        
        # 회귀 메트릭
        mae = mean_absolute_error(y_true, y_pred)
        mse = np.mean((y_true - y_pred)**2)
        rmse = np.sqrt(mse)
        
        # 분류 정확도 (±3 오차 허용)
        tolerance = 3
        correct_predictions = np.abs(y_true - y_pred) <= tolerance
        accuracy = np.mean(correct_predictions)
        
        print(f"\n=== {dataset_name} 평가 결과 (CNN+LSTM Hybrid) ===")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Square Error: {rmse:.2f}")
        print(f"Accuracy (±{tolerance} tolerance): {accuracy:.3f}")
        print(f"Prediction range: {y_pred.min():.1f} - {y_pred.max():.1f} psi")
        
        # 각 클래스별 정확도
        for pressure in [35, 50, 60]:
            mask = y_true == pressure
            if np.sum(mask) > 0:
                class_mae = np.mean(np.abs(y_pred[mask] - y_true[mask]))
                class_accuracy = np.mean(np.abs(y_pred[mask] - y_true[mask]) <= tolerance)
                print(f"{pressure}psi - MAE: {class_mae:.2f}, Accuracy: {class_accuracy:.3f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    def plot_results(self, history, train_results=None, val_results=None):
        """결과 시각화"""
        if train_results is None or val_results is None:
            train_results, val_results = self.evaluate_on_train_val()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('CNN+LSTM Hybrid Model Results', fontsize=16)
        
        # 훈련 히스토리
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        axes[0, 1].plot(history.history['mae'], label='Training MAE')
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        
        # 훈련 데이터 예측 vs 실제
        if train_results:
            y_train = self.train_data[2]
            y_pred_train = train_results['predictions']
            axes[0, 2].scatter(y_train, y_pred_train, alpha=0.6, label='Training')
            axes[0, 2].plot([32, 64], [32, 64], 'r--', lw=2)
            axes[0, 2].set_xlabel('True Pressure')
            axes[0, 2].set_ylabel('Predicted Pressure')
            axes[0, 2].set_title('Training: Predictions vs True Values')
            axes[0, 2].set_xlim([32, 64])
            axes[0, 2].set_ylim([32, 64])
            axes[0, 2].legend()
        
        # 검증 데이터 예측 vs 실제
        if val_results:
            y_val = self.val_data[2]
            y_pred_val = val_results['predictions']
            axes[1, 0].scatter(y_val, y_pred_val, alpha=0.6, color='orange', label='Validation')
            axes[1, 0].plot([32, 64], [32, 64], 'r--', lw=2)
            axes[1, 0].set_xlabel('True Pressure')
            axes[1, 0].set_ylabel('Predicted Pressure')
            axes[1, 0].set_title('Validation: Predictions vs True Values')
            axes[1, 0].set_xlim([32, 64])
            axes[1, 0].set_ylim([32, 64])
            axes[1, 0].legend()
        
        # 훈련 데이터 오차 분포
        if train_results:
            errors_train = y_pred_train - y_train
            axes[1, 1].hist(errors_train, bins=30, alpha=0.7, label='Training')
            axes[1, 1].axvline(x=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Prediction Error')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Training Error Distribution')
            axes[1, 1].legend()
        
        # 검증 데이터 오차 분포
        if val_results:
            errors_val = y_pred_val - y_val
            axes[1, 2].hist(errors_val, bins=30, alpha=0.7, color='orange', label='Validation')
            axes[1, 2].axvline(x=0, color='r', linestyle='--')
            axes[1, 2].set_xlabel('Prediction Error')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Validation Error Distribution')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()

def main():
    print("=== 타이어 공기압 예측 CNN+LSTM 하이브리드 모델 ===")
    print("- 예측 범위: 32-64 psi")
    print("- CNN: 공간적 특성 추출")
    print("- LSTM: 시간적 의존성 학습")
    print("- 최고 성능을 위한 하이브리드 구조")
    print("Google Colab 버전")
    print()
    
    # 모델 초기화
    predictor = TirePressureHybridPredictor(sequence_length=1000, step_size=100)
    
    # 데이터 로드
    print("Loading data...")
    X_raw, y = predictor.load_data()
    print(f"Raw data shape: {X_raw.shape}")
    print(f"Labels shape: {y.shape}")
    
    # 라벨 분포 확인
    unique, counts = np.unique(y, return_counts=True)
    print(f"Label distribution:")
    for pressure, count in zip(unique, counts):
        print(f"  {pressure}psi: {count} sequences")
    
    # 특성 추출
    print("\nExtracting features...")
    X_time_features, X_freq_features = predictor.extract_all_features(X_raw)
    X_features = np.concatenate([X_time_features, X_freq_features], axis=1)
    print(f"Time domain features shape: {X_time_features.shape}")
    print(f"Frequency domain features shape: {X_freq_features.shape}")
    print(f"Combined features shape: {X_features.shape}")
    
    # 모델 훈련
    print("\nTraining CNN+LSTM Hybrid model...")
    history = predictor.train(X_raw, X_features, y, validation_split=0.1, epochs=50, batch_size=32)
    
    # 훈련 및 검증 데이터에 대한 평가
    print("\nEvaluating on training and validation data...")
    train_results, val_results = predictor.evaluate_on_train_val()
    
    # 결과 시각화
    predictor.plot_results(history, train_results, val_results)
    
    # 모델 저장
    predictor.model.save('tire_pressure_hybrid_model.h5')
    print("\nModel saved as 'tire_pressure_hybrid_model.h5'")
    
    # 모델 다운로드
    print("\n모델 파일을 다운로드하시겠습니까?")
    download_choice = input("y/n: ").lower()
    if download_choice == 'y':
        files.download('tire_pressure_hybrid_model.h5')
    
    return predictor, train_results, val_results

# 실행
if __name__ == "__main__":
    predictor, train_results, val_results = main()

print("\n=== CNN+LSTM 하이브리드 모델 특징 ===")
print("✅ CNN: 지역적 패턴과 공간적 특성 추출")
print("✅ LSTM: 시간적 의존성과 순차적 패턴 학습")
print("✅ 다중 브랜치: 다양한 관점에서 특성 학습")
print("✅ 최고 성능: CNN과 LSTM의 장점 결합") 