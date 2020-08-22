from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class ImputarMediana(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns = columns

    def fit(self, X, y=None):
        import numpy as np
        import pandas as pd
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        import numpy as np
        import pandas as pd
        data = X.copy()
        numeric_columns = data.select_dtypes(exclude='object').columns
        
        for columna in numeric_columns:
            mediana = data[columna].dropna().median()
            data[columna].fillna(mediana,inplace=True)
        return data
    
class Estandar_Data(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        numeric_columns = data.select_dtypes(exclude='object').columns
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        return data
