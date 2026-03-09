import pytest
import pandas as pd
from src.preprocessing import (
    normalize_df_columns,
    create_target_column,
    transform_instituicao_text
)
 
 
class TestNormalizeDfColumns:
    """Testes para a função normalize_df_columns"""
    
    def test_normalize_lowercase(self):
        df = pd.DataFrame({"NOME": [1], "IDADE": [2]})
        result = normalize_df_columns(df)
        assert list(result.columns) == ["nome", "idade"]
    
    def test_normalize_remove_accents(self):
        df = pd.DataFrame({"São Paulo": [1], "Açúcar": [2]})
        result = normalize_df_columns(df)
        assert list(result.columns) == ["sao_paulo", "acucar"]
    
    def test_normalize_combined(self):
        df = pd.DataFrame({
            "Nome Completo": [1],
            "ENDEREÇO": [2],
            "Número De Filhos": [3]
        })
        result = normalize_df_columns(df)
        expected_columns = [
            "nome_completo",
            "endereco",
            "numero_de_filhos"
        ]
        assert list(result.columns) == expected_columns
 
 
class TestCreateTargetColumn:
    """Testes para a função create_target_column"""
    
    def test_create_target_negative_values(self):
        df = pd.DataFrame({"defas": [-1, -5, -10]})
        result = create_target_column(df)
        assert result["target"].tolist() == [1, 1, 1]
    
    def test_create_target_positive_values(self):
        df = pd.DataFrame({"defas": [1, 5, 10]})
        result = create_target_column(df)
        assert result["target"].tolist() == [0, 0, 0]
    
    def test_create_target_mixed_values(self):
        df = pd.DataFrame({"defas": [-5, -1, 0, 1, 5]})
        result = create_target_column(df)
        assert result["target"].tolist() == [1, 1, 0, 0, 0]
 
 
class TestTransformInstituicaoText:
    """Testes para a função transform_instituicao_text"""
    
    def test_transform_escola_publica(self):
        df = pd.DataFrame({
            "instituicao_de_ensino": ["Escola Pública", "Escola Pública"]
        })
        result = transform_instituicao_text(df)
        assert result["instituicao_de_ensino"].tolist() == [
            "escola publica", "escola publica"
        ]
    
    def test_transform_escola_privada(self):
        df = pd.DataFrame({
            "instituicao_de_ensino": ["Escola Privada", "Colégio Particular"]
        })
        result = transform_instituicao_text(df)
        assert result["instituicao_de_ensino"].tolist() == [
            "escola privada", "escola privada"
        ]
    
    def test_transform_mixed_values(self):
        df = pd.DataFrame({
            "instituicao_de_ensino": [
                "Escola Pública",
                "Escola Privada",
                "Escola Pública",
                "Colégio XYZ"
            ]
        })
        result = transform_instituicao_text(df)
        expected = [
            "escola publica",
            "escola privada",
            "escola publica",
            "escola privada"
        ]
        assert result["instituicao_de_ensino"].tolist() == expected


if __name__ == "__main__":

    pytest.main([__file__, "-v"])