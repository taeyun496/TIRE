# Google Colab용 타이어 공기압 예측 XGBoost 모델
# 사용법: Google Colab에서 이 코드를 복사하여 실행하세요

# 필요한 패키지 설치
# !pip install xgboost numpy pandas scikit-learn matplotlib seaborn scipy

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
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
import pickle

warnings.filterwarnings('ignore')

class TirePressureXGBoostPredictor:
    def __init__(self, sequence_length=1000, step_size=100):
        self.sequence_length = sequence_length
        self.step_size = step_size
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
        """시간 도메인 특성 추출 (확장된 버전)"""
        features = []
        
        for channel in range(sequence.shape[1]):  # 6개 채널
            data = sequence[:, channel]
            
            # 기본 통계량
            features.extend([
                np.mean(data),           # 평균
                np.std(data),            # 표준편차
                np.var(data),            # 분산
                np.max(data),            # 최대값
                np.min(data),            # 최소값
                np.sqrt(np.mean(data**2)), # RMS
                skew(data),              # 왜도
                kurtosis(data),          # 첨도
                np.ptp(data),            # Peak-to-peak
                np.percentile(data, 25), # 1사분위수
                np.percentile(data, 50), # 중앙값
                np.percentile(data, 75), # 3사분위수
                np.percentile(data, 10), # 10분위수
                np.percentile(data, 90), # 90분위수
            ])
            
            # 추가 통계량
            features.extend([
                np.mean(np.abs(data)),   # 절대값 평균
                np.mean(data**2),        # 제곱 평균
                np.sum(np.abs(np.diff(data))), # 총 변화량
                np.std(np.diff(data)),   # 변화율 표준편차
                len(data[data > np.mean(data)]) / len(data), # 평균 이상 비율
            ])
            
            # 영교차 횟수
            zero_crossings = np.sum(np.diff(np.sign(data - np.mean(data))) != 0)
            features.append(zero_crossings)
            
            # 피크 개수
            peaks, _ = signal.find_peaks(data)
            features.append(len(peaks))
            
            # 에너지
            energy = np.sum(data**2)
            features.append(energy)
        
        return np.array(features)
    
    def extract_frequency_domain_features(self, sequence, fs=28.6):
        """주파수 도메인 특성 추출 (확장된 버전)"""
        features = []
        
        for channel in range(sequence.shape[1]):  # 6개 채널
            data = sequence[:, channel]
            
            # FFT 계산
            fft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data), 1/fs)
            
            # 양의 주파수만 사용
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # 주파수 대역별 파워 (더 세분화)
            freq_bands = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 15)]
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
            
            # 스펙트럼 대역폭
            spectral_bandwidth = np.sqrt(np.sum(((positive_freqs - spectral_centroid)**2) * positive_fft) / np.sum(positive_fft))
            features.append(spectral_bandwidth)
            
            # 스펙트럼 평탄도
            geometric_mean = np.exp(np.mean(np.log(positive_fft + 1e-10)))
            arithmetic_mean = np.mean(positive_fft)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
            features.append(spectral_flatness)
            
            # 스펙트럼 기울기
            freqs_log = np.log(positive_freqs + 1e-10)
            fft_log = np.log(positive_fft + 1e-10)
            spectral_slope = np.polyfit(freqs_log, fft_log, 1)[0]
            features.append(spectral_slope)
        
        return np.array(features)
    
    def extract_cross_channel_features(self, sequence):
        """채널 간 상관관계 특성 추출"""
        features = []
        
        # 채널 간 상관계수
        corr_matrix = np.corrcoef(sequence.T)
        # 상삼각 행렬의 값들만 추출 (대각선 제외)
        upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
        correlations = corr_matrix[upper_tri_indices]
        features.extend(correlations)
        
        # 채널 간 공분산
        cov_matrix = np.cov(sequence.T)
        covariances = cov_matrix[upper_tri_indices]
        features.extend(covariances)
        
        # 주성분 분석 특성
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(sequence)
        
        # 주성분의 분산 비율
        features.extend(pca.explained_variance_ratio_)
        
        # 각 주성분의 통계량
        for i in range(3):
            pc = pca_result[:, i]
            features.extend([
                np.mean(pc),
                np.std(pc),
                skew(pc),
                kurtosis(pc)
            ])
        
        return np.array(features)
    
    def extract_all_features(self, sequences):
        """모든 특성 추출"""
        time_features = []
        freq_features = []
        cross_features = []
        
        print("Extracting features for XGBoost...")
        for i, seq in enumerate(sequences):
            if i % 100 == 0:
                print(f"Processing sequence {i}/{len(sequences)}")
            
            time_feat = self.extract_time_domain_features(seq)
            freq_feat = self.extract_frequency_domain_features(seq)
            cross_feat = self.extract_cross_channel_features(seq)
            
            time_features.append(time_feat)
            freq_features.append(freq_feat)
            cross_features.append(cross_feat)
        
        return np.array(time_features), np.array(freq_features), np.array(cross_features)
    
    def train(self, X_features, y, validation_split=0.1, use_grid_search=True):
        """XGBoost 모델 훈련"""
        # 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X_features, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training data: {len(X_train)} samples")
        print(f"Validation data: {len(X_val)} samples")
        print(f"Training ratio: {len(X_train)/(len(X_train)+len(X_val)):.1%}")
        print(f"Feature dimensions: {X_features.shape[1]}")
        
        # 특성 정규화
        X_train_scaled = self.scaler_features.fit_transform(X_train)
        X_val_scaled = self.scaler_features.transform(X_val)
        
        if use_grid_search:
            print("Performing Grid Search for hyperparameter tuning...")
            # 하이퍼파라미터 그리드 서치
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                xgb_model, 
                param_grid, 
                cv=3, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.3f}")
            
        else:
            # 기본 파라미터로 훈련
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        
        # 훈련 및 검증 데이터 저장
        self.train_data = (X_train_scaled, y_train)
        self.val_data = (X_val_scaled, y_val)
        
        return self.model
    
    def predict(self, X_features):
        """예측 (32-64 psi 범위로 제한)"""
        X_scaled = self.scaler_features.transform(X_features)
        predictions = self.model.predict(X_scaled)
        
        # 예측값을 32-64 범위로 클리핑
        predictions = np.clip(predictions, self.min_pressure, self.max_pressure)
        
        return predictions
    
    def evaluate_model(self, X_features, y_true, dataset_name="Dataset"):
        """모델 평가"""
        y_pred = self.predict(X_features)
        
        # 회귀 메트릭
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # 분류 정확도 (±3 오차 허용)
        tolerance = 3
        correct_predictions = np.abs(y_true - y_pred) <= tolerance
        accuracy = np.mean(correct_predictions)
        
        print(f"\n=== {dataset_name} 평가 결과 (XGBoost) ===")
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
        X_train_scaled, y_train = self.train_data
        train_results = self.evaluate_model_direct(X_train_scaled, y_train, "Training")
        
        # 검증 데이터 평가
        X_val_scaled, y_val = self.val_data
        val_results = self.evaluate_model_direct(X_val_scaled, y_val, "Validation")
        
        return train_results, val_results
    
    def evaluate_model_direct(self, X_scaled, y_true, dataset_name="Dataset"):
        """이미 스케일링된 데이터에 대한 직접 평가"""
        predictions = self.model.predict(X_scaled)
        y_pred = np.clip(predictions, self.min_pressure, self.max_pressure)
        
        # 회귀 메트릭
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # 분류 정확도 (±3 오차 허용)
        tolerance = 3
        correct_predictions = np.abs(y_true - y_pred) <= tolerance
        accuracy = np.mean(correct_predictions)
        
        print(f"\n=== {dataset_name} 평가 결과 (XGBoost) ===")
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
    
    def plot_feature_importance(self, top_n=20):
        """특성 중요도 시각화"""
        if self.model is None:
            print("모델이 훈련되지 않았습니다.")
            return
        
        # 특성 중요도 가져오기
        importance = self.model.feature_importances_
        
        # 특성 이름 생성
        feature_names = self.generate_feature_names()
        
        # 중요도 순으로 정렬
        indices = np.argsort(importance)[::-1]
        
        # 상위 N개 특성 시각화
        plt.figure(figsize=(12, 8))
        plt.title(f"Top {top_n} Feature Importance (XGBoost)")
        plt.bar(range(top_n), importance[indices[:top_n]])
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # 상위 특성 출력
        print(f"\nTop {top_n} most important features:")
        for i in range(top_n):
            idx = indices[i]
            print(f"{i+1:2d}. {feature_names[idx]:40s}: {importance[idx]:.4f}")
    
    def generate_feature_names(self):
        """특성 이름 생성"""
        feature_names = []
        channels = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']
        
        # 시간 도메인 특성
        time_features = ['mean', 'std', 'var', 'max', 'min', 'rms', 'skew', 'kurtosis', 'ptp', 
                        'q25', 'q50', 'q75', 'q10', 'q90', 'abs_mean', 'sq_mean', 'total_var', 
                        'var_std', 'above_mean_ratio', 'zero_crossings', 'peaks', 'energy']
        
        for channel in channels:
            for feat in time_features:
                feature_names.append(f'{channel}_{feat}')
        
        # 주파수 도메인 특성
        freq_features = ['band_0-1Hz', 'band_1-2Hz', 'band_2-3Hz', 'band_3-5Hz', 'band_5-7Hz', 
                        'band_7-10Hz', 'band_10-15Hz', 'dominant_freq', 'spectral_centroid', 
                        'total_power', 'spectral_rolloff', 'spectral_bandwidth', 'spectral_flatness', 
                        'spectral_slope']
        
        for channel in channels:
            for feat in freq_features:
                feature_names.append(f'{channel}_{feat}')
        
        # 채널 간 특성
        cross_features = []
        for i in range(len(channels)):
            for j in range(i+1, len(channels)):
                cross_features.append(f'corr_{channels[i]}_{channels[j]}')
                cross_features.append(f'cov_{channels[i]}_{channels[j]}')
        
        cross_features.extend(['pca_var_ratio_1', 'pca_var_ratio_2', 'pca_var_ratio_3'])
        
        for i in range(3):
            cross_features.extend([f'pca{i+1}_mean', f'pca{i+1}_std', f'pca{i+1}_skew', f'pca{i+1}_kurtosis'])
        
        feature_names.extend(cross_features)
        
        return feature_names
    
    def plot_results(self, train_results=None, val_results=None):
        """결과 시각화"""
        if train_results is None or val_results is None:
            train_results, val_results = self.evaluate_on_train_val()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('XGBoost Model Results', fontsize=16)
        
        # 특성 중요도
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            indices = np.argsort(importance)[::-1][:15]
            feature_names = self.generate_feature_names()
            
            axes[0, 0].bar(range(15), importance[indices])
            axes[0, 0].set_title('Top 15 Feature Importance')
            axes[0, 0].set_xlabel('Features')
            axes[0, 0].set_ylabel('Importance')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 훈련 데이터 예측 vs 실제
        if train_results:
            y_train = self.train_data[1]
            y_pred_train = train_results['predictions']
            axes[0, 1].scatter(y_train, y_pred_train, alpha=0.6, label='Training')
            axes[0, 1].plot([32, 64], [32, 64], 'r--', lw=2)
            axes[0, 1].set_xlabel('True Pressure')
            axes[0, 1].set_ylabel('Predicted Pressure')
            axes[0, 1].set_title('Training: Predictions vs True Values')
            axes[0, 1].set_xlim([32, 64])
            axes[0, 1].set_ylim([32, 64])
            axes[0, 1].legend()
        
        # 검증 데이터 예측 vs 실제
        if val_results:
            y_val = self.val_data[1]
            y_pred_val = val_results['predictions']
            axes[0, 2].scatter(y_val, y_pred_val, alpha=0.6, color='orange', label='Validation')
            axes[0, 2].plot([32, 64], [32, 64], 'r--', lw=2)
            axes[0, 2].set_xlabel('True Pressure')
            axes[0, 2].set_ylabel('Predicted Pressure')
            axes[0, 2].set_title('Validation: Predictions vs True Values')
            axes[0, 2].set_xlim([32, 64])
            axes[0, 2].set_ylim([32, 64])
            axes[0, 2].legend()
        
        # 훈련 데이터 오차 분포
        if train_results:
            errors_train = y_pred_train - y_train
            axes[1, 0].hist(errors_train, bins=30, alpha=0.7, label='Training')
            axes[1, 0].axvline(x=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Prediction Error')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Training Error Distribution')
            axes[1, 0].legend()
        
        # 검증 데이터 오차 분포
        if val_results:
            errors_val = y_pred_val - y_val
            axes[1, 1].hist(errors_val, bins=30, alpha=0.7, color='orange', label='Validation')
            axes[1, 1].axvline(x=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Prediction Error')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Validation Error Distribution')
            axes[1, 1].legend()
        
        # 압력별 정확도 비교
        if train_results and val_results:
            pressures = [35, 50, 60]
            train_accuracies = []
            val_accuracies = []
            
            for pressure in pressures:
                # 훈련 데이터 정확도
                train_mask = y_train == pressure
                if np.sum(train_mask) > 0:
                    train_acc = np.mean(np.abs(y_pred_train[train_mask] - y_train[train_mask]) <= 3)
                    train_accuracies.append(train_acc)
                else:
                    train_accuracies.append(0)
                
                # 검증 데이터 정확도
                val_mask = y_val == pressure
                if np.sum(val_mask) > 0:
                    val_acc = np.mean(np.abs(y_pred_val[val_mask] - y_val[val_mask]) <= 3)
                    val_accuracies.append(val_acc)
                else:
                    val_accuracies.append(0)
            
            x = np.arange(len(pressures))
            width = 0.35
            
            axes[1, 2].bar(x - width/2, train_accuracies, width, label='Training', alpha=0.8)
            axes[1, 2].bar(x + width/2, val_accuracies, width, label='Validation', alpha=0.8)
            
            axes[1, 2].set_xlabel('Pressure (psi)')
            axes[1, 2].set_ylabel('Accuracy (±3 psi)')
            axes[1, 2].set_title('Accuracy by Pressure Level')
            axes[1, 2].set_xticks(x)
            axes[1, 2].set_xticklabels(pressures)
            axes[1, 2].legend()
            axes[1, 2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()

def main():
    print("=== 타이어 공기압 예측 XGBoost 모델 ===")
    print("- 예측 범위: 32-64 psi")
    print("- 특성 기반 머신러닝 모델")
    print("- 빠른 훈련과 높은 해석 가능성")
    print("Google Colab 버전")
    print()
    
    # 모델 초기화
    predictor = TirePressureXGBoostPredictor(sequence_length=1000, step_size=100)
    
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
    print("\nExtracting comprehensive features for XGBoost...")
    X_time_features, X_freq_features, X_cross_features = predictor.extract_all_features(X_raw)
    X_features = np.concatenate([X_time_features, X_freq_features, X_cross_features], axis=1)
    print(f"Time domain features shape: {X_time_features.shape}")
    print(f"Frequency domain features shape: {X_freq_features.shape}")
    print(f"Cross-channel features shape: {X_cross_features.shape}")
    print(f"Total features shape: {X_features.shape}")
    
    # 모델 훈련
    print("\nTraining XGBoost model...")
    use_grid_search = input("하이퍼파라미터 그리드 서치를 수행하시겠습니까? (y/n): ").lower() == 'y'
    model = predictor.train(X_features, y, validation_split=0.1, use_grid_search=use_grid_search)
    
    # 훈련 및 검증 데이터에 대한 평가
    print("\nEvaluating on training and validation data...")
    train_results, val_results = predictor.evaluate_on_train_val()
    
    # 특성 중요도 시각화
    predictor.plot_feature_importance(top_n=20)
    
    # 결과 시각화
    predictor.plot_results(train_results, val_results)
    
    # 모델 저장
    with open('tire_pressure_xgboost_model.pkl', 'wb') as f:
        pickle.dump(predictor, f)
    print("\nModel saved as 'tire_pressure_xgboost_model.pkl'")
    
    # 모델 다운로드
    print("\n모델 파일을 다운로드하시겠습니까?")
    download_choice = input("y/n: ").lower()
    if download_choice == 'y':
        files.download('tire_pressure_xgboost_model.pkl')
    
    return predictor, train_results, val_results

# 실행
if __name__ == "__main__":
    predictor, train_results, val_results = main()

print("\n=== XGBoost 모델 특징 ===")
print("✅ 특성 기반 학습: 시간/주파수/채널간 특성 활용")
print("✅ 빠른 훈련: 딥러닝 대비 훨씬 빠른 학습 속도")
print("✅ 해석 가능성: 특성 중요도 분석 가능")
print("✅ 하이퍼파라미터 최적화: 그리드 서치 지원") 