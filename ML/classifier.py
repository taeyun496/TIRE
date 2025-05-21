import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from scipy import signal
import os
from google.colab import files
import io
import joblib
import pickle

class PressureClassifier:
    def __init__(self):
        """
        압력 분류기 초기화
        - 샘플링 간격: 35ms
        - 목표 압력: 35psi, 50psi, 60psi
        """
        self.sampling_interval = 0.035
        self.sampling_freq = 1 / self.sampling_interval
        self.pressure_levels = [35, 50, 60]
        self.scaler = StandardScaler()
        self.models = None
        
    def extract_features(self, data):
        """
        시계열 데이터에서 특성 추출
        
        Parameters:
        - data: 센서 데이터 (DataFrame)
        
        Returns:
        - features: 추출된 특성 (numpy array)
        """
        features = []
        
        # 분석할 센서 컬럼들
        columns = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']
        
        for column in columns:
            # 시간 도메인 특성
            signal_data = data[column].values
            
            # 기본 통계 특성
            features.extend([
                np.mean(signal_data),
                np.std(signal_data),
                np.max(signal_data),
                np.min(signal_data),
                np.median(signal_data)
            ])
            
            # 주파수 도메인 특성
            freqs, psd = signal.welch(signal_data, 
                                     fs=self.sampling_freq,
                                     nperseg=256,
                                     noverlap=128)
            
            # 주파수 대역별 에너지
            freq_bands = [(0, 5), (5, 10), (10, 15)]  # Hz
            for low, high in freq_bands:
                mask = (freqs >= low) & (freqs < high)
                band_energy = np.sum(psd[mask])
                features.append(band_energy)
        
        return np.array(features)
    
    def prepare_dataset(self, data_files, pressure_labels):
        """
        데이터셋 준비
        
        Parameters:
        - data_files: 데이터 파일 경로 리스트
        - pressure_labels: 각 파일에 해당하는 압력 레이블
        
        Returns:
        - X: 특성 행렬
        - y: 압력 레이블
        """
        X = []
        y = []
        
        for file_path, pressure in zip(data_files, pressure_labels):
            # CSV 파일 로드
            df = pd.read_csv(file_path)
            
            # 특성 추출
            features = self.extract_features(df)
            X.append(features)
            y.append(pressure)
        
        X = np.array(X)
        y = np.array(y)
        
        # 특성 정규화
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train_models(self, X, y):
        """
        여러 모델 학습 및 앙상블
        
        Parameters:
        - X: 특성 행렬
        - y: 압력 레이블
        
        Returns:
        - models: 학습된 모델 리스트
        """
        # 기본 분류기들
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        
        # 모델 학습
        rf_model.fit(X, y)
        xgb_model.fit(X, y)
        
        return [rf_model, xgb_model]
    
    def evaluate_models(self, models, X_test, y_test):
        """
        모델 평가
        
        Parameters:
        - models: 학습된 모델 리스트
        - X_test: 테스트 특성
        - y_test: 테스트 레이블
        """
        for i, model in enumerate(models):
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nModel {i+1} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=[f"{p}psi" for p in self.pressure_levels]))
    
    def cross_validate(self, X, y, n_splits=5):
        """
        K-fold 교차 검증
        
        Parameters:
        - X: 특성 행렬
        - y: 압력 레이블
        - n_splits: 교차 검증 폴드 수
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 모델 학습
            models = self.train_models(X_train, y_train)
            
            # 앙상블 예측
            predictions = []
            for model in models:
                pred = model.predict_proba(X_val)
                predictions.append(pred)
            
            # 평균 예측
            ensemble_pred = np.mean(predictions, axis=0)
            y_pred = np.argmax(ensemble_pred, axis=1)
            y_pred = np.array([self.pressure_levels[i] for i in y_pred])
            
            # 정확도 계산
            accuracy = accuracy_score(y_val, y_pred)
            fold_scores.append(accuracy)
            
            print(f"\nFold {fold+1} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_val, y_pred, target_names=[f"{p}psi" for p in self.pressure_levels]))
        
        print(f"\nAverage Accuracy across {n_splits} folds: {np.mean(fold_scores):.4f}")

    def save_models(self, save_dir='models'):
        """
        학습된 모델과 스케일러 저장
        
        Parameters:
        - save_dir: 모델을 저장할 디렉토리
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 스케일러 저장
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # 모델 저장
        for i, model in enumerate(self.models):
            if isinstance(model, RandomForestClassifier):
                model_path = os.path.join(save_dir, f'rf_model_{i}.pkl')
            else:  # XGBoost
                model_path = os.path.join(save_dir, f'xgb_model_{i}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
        print(f"모델이 {save_dir} 디렉토리에 저장되었습니다.")
        
        # Google Colab에서 파일 다운로드
        for file in os.listdir(save_dir):
            files.download(os.path.join(save_dir, file))
    
    def load_models(self, model_dir='models'):
        """
        저장된 모델과 스케일러 로드
        
        Parameters:
        - model_dir: 모델이 저장된 디렉토리
        """
        # 스케일러 로드
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
            
        # 모델 로드
        self.models = []
        for file in os.listdir(model_dir):
            if file.endswith('.pkl') and file != 'scaler.pkl':
                with open(os.path.join(model_dir, file), 'rb') as f:
                    self.models.append(pickle.load(f))
                    
        print(f"{len(self.models)}개의 모델이 로드되었습니다.")
    
    def predict_pressure(self, data):
        """
        새로운 데이터에 대한 압력 예측
        
        Parameters:
        - data: 예측할 센서 데이터 (DataFrame)
        
        Returns:
        - predicted_pressure: 예측된 압력 값
        - confidence: 예측 신뢰도
        """
        if self.models is None:
            raise ValueError("모델이 로드되지 않았습니다. load_models()를 먼저 실행하세요.")
            
        # 특성 추출
        features = self.extract_features(data)
        features = self.scaler.transform(features.reshape(1, -1))
        
        # 각 모델의 예측 확률
        predictions = []
        for model in self.models:
            pred = model.predict_proba(features)
            predictions.append(pred)
            
        # 앙상블 예측
        ensemble_pred = np.mean(predictions, axis=0)
        predicted_idx = np.argmax(ensemble_pred)
        predicted_pressure = self.pressure_levels[predicted_idx]
        confidence = ensemble_pred[0][predicted_idx]
        
        return predicted_pressure, confidence

def main():
    """
    메인 실행 함수
    """
    # 분류기 초기화
    classifier = PressureClassifier()
    
    # 데이터 파일 업로드
    print("압력별 데이터 파일을 업로드하세요...")
    uploaded = files.upload()
    
    if not uploaded:
        print("파일이 업로드되지 않았습니다.")
        return
    
    # 파일 정렬 및 레이블 생성
    data_files = []
    pressure_labels = []
    
    for filename in uploaded.keys():
        # 파일명에서 압력 정보 추출 (예: mpu_data_35psi_001.csv)
        pressure = int(filename.split('_')[2].replace('psi', ''))
        data_files.append(filename)
        pressure_labels.append(pressure)
    
    # 데이터셋 준비
    X, y = classifier.prepare_dataset(data_files, pressure_labels)
    
    # 데이터 분할 (3:1:1)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 모델 학습
    classifier.models = classifier.train_models(X_train, y_train)
    
    # 모델 평가
    print("\nValidation Set Results:")
    classifier.evaluate_models(classifier.models, X_val, y_val)
    
    print("\nTest Set Results:")
    classifier.evaluate_models(classifier.models, X_test, y_test)
    
    # 교차 검증
    print("\nCross-validation Results:")
    classifier.cross_validate(X, y)
    
    # 모델 저장
    classifier.save_models()

if __name__ == "__main__":
    main() 
