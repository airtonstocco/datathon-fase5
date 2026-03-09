import sys
import os
import pytest
import numpy as np
import pandas as pd
import joblib
from unittest.mock import patch, MagicMock
from src.utils import balance_threshold, oversampling
from src.evaluate import evaluate_model
from src import train as train_module
 
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
 
 
@pytest.fixture
def sample_binary_classification_data():
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100)
    })
    y = pd.Series(np.random.choice([0, 1], 100), name='target')
    return X, y
 
 
@pytest.fixture
def sample_imbalanced_data():
    X = pd.DataFrame({
        'feature1': list(range(100)),
        'feature2': list(range(100, 200))
    })
    y = pd.Series([0] * 80 + [1] * 20, name='target')
    return X, y
 
 
@pytest.fixture
def sample_predictions():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.4, 0.2, 0.85, 0.15])
    return y_true, y_pred, y_prob
 
 
@pytest.fixture
def mock_dataframe():
    np.random.seed(42)
    return pd.DataFrame({
        'NOME': ['A'] * 200,
        'instituicao_de_ensino': ['Escola Pública'] * 100 + ['Escola Privada'] * 100,
        'pedra_20': ['Sim'] * 200,
        'pedra_21': ['Não'] * 200,
        'pedra_22': ['Sim'] * 200,
        'indicado': ['Sim'] * 200,
        'atingiu_pv': ['Não'] * 200,
        'cg': np.random.rand(200) * 10,
        'cf': np.random.rand(200) * 10,
        'ct': np.random.rand(200) * 10,
        'inde_22': np.random.rand(200) * 100,
        'defas': np.random.choice([-1, 1], 200)
    })
 
 
class TestBalanceThreshold:
    def test_threshold_in_valid_range(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        threshold = balance_threshold(y_true, y_prob)
        assert 0 <= threshold <= 1
 
    def test_threshold_is_float(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        threshold = balance_threshold(y_true, y_prob)
        assert isinstance(threshold, (float, np.floating))
 
 
class TestOversampling:
    def test_balances_classes(self):
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [10, 20, 30, 40, 50, 60]
        })
        y_train = pd.Series([0, 0, 0, 0, 1, 1], name='target')
        result = oversampling(X_train, y_train)
        assert result[result.target == 0].shape[0] == result[result.target == 1].shape[0]
 
    def test_returns_dataframe(self):
        X_train = pd.DataFrame({'feature': [1, 2, 3, 4]})
        y_train = pd.Series([0, 0, 1, 1], name='target')
        result = oversampling(X_train, y_train)
        assert isinstance(result, pd.DataFrame)
 
    def test_preserves_columns(self):
        X_train = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [10, 20, 30]
        })
        y_train = pd.Series([0, 1, 1], name='target')
        result = oversampling(X_train, y_train)
        expected_columns = ['col1', 'col2', 'target']
        assert all(col in result.columns for col in expected_columns)
 
    def test_minority_class_unchanged(self):
        X_train = pd.DataFrame({'feature': [1, 2, 3, 4]})
        y_train = pd.Series([0, 0, 0, 1], name='target')
        result = oversampling(X_train, y_train)
        class_1_values = result[result.target == 1]['feature'].values
        assert 4 in class_1_values
 
 
class TestEvaluateModel:
    def test_returns_correct_structure(self):
        y_test = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.2, 0.6, 0.8, 0.9])
        result = evaluate_model(0.5, y_test, y_pred, y_prob)
        assert 'threshold' in result
        assert 'metrics' in result
        assert 'confusion_matrix' in result
 
    def test_metrics_in_valid_range(self):
        y_test = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4])
        result = evaluate_model(0.5, y_test, y_pred, y_prob)
        metrics = result['metrics']
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"Métrica '{metric_name}' fora do intervalo: {value}"
 
    def test_confusion_matrix_sums_correctly(self):
        y_test = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.4, 0.2])
        result = evaluate_model(0.5, y_test, y_pred, y_prob)
        cm = result['confusion_matrix']
        total = cm['true_negative'] + cm['false_positive'] + \
                cm['false_negative'] + cm['true_positive']
        assert total == len(y_test)
 
    def test_perfect_predictions(self):
        y_test = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.95])
        result = evaluate_model(0.5, y_test, y_pred, y_prob)
        metrics = result['metrics']
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
 
    def test_confusion_matrix_values_are_integers(self):
        y_test = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.2, 0.8, 0.6, 0.9])
        result = evaluate_model(0.5, y_test, y_pred, y_prob)
        cm = result['confusion_matrix']
        for key, value in cm.items():
            assert isinstance(value, int), f"'{key}' não é inteiro: {type(value)}"
 
 
class TestTrainModelPipeline:
    def test_pipeline_runs_successfully(self, mock_dataframe):
        with patch('src.train.pd.read_excel', return_value=mock_dataframe):
            result = train_module.train_model_pipeline()
        assert 'status' in result
        assert 'evaluation' in result
        assert result['status'] == 'Modelo treinado com sucesso'
 
    def test_model_file_is_created(self, mock_dataframe):
        if os.path.exists('model.pkl'):
            os.remove('model.pkl')
        with patch('src.train.pd.read_excel', return_value=mock_dataframe):
            train_module.train_model_pipeline()
        assert os.path.exists('model.pkl')
        if os.path.exists('model.pkl'):
            os.remove('model.pkl')
 
    def test_saved_model_has_correct_keys(self, mock_dataframe):
        with patch('src.train.pd.read_excel', return_value=mock_dataframe):
            train_module.train_model_pipeline()
        saved = joblib.load('model.pkl')
        assert 'model' in saved
        assert 'threshold' in saved
        assert 'features' in saved
        assert hasattr(saved['model'], 'predict')
        assert isinstance(saved['threshold'], (float, np.floating))
        assert isinstance(saved['features'], list)
        if os.path.exists('model.pkl'):
            os.remove('model.pkl')
 
    def test_evaluation_contains_all_metrics(self, mock_dataframe):
        with patch('src.train.pd.read_excel', return_value=mock_dataframe):
            result = train_module.train_model_pipeline()
        evaluation = result['evaluation']
        metrics = evaluation['metrics']
        assert 'threshold' in evaluation
        assert 'confusion_matrix' in evaluation
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        if os.path.exists('model.pkl'):
            os.remove('model.pkl')
 
 
if __name__ == "__main__":
    pytest.main([__file__, "-v"])