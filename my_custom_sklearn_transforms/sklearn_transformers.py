from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

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
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        numeric_columns = data.select_dtypes(exclude='object').columns
        
        from sklearn.impute import SimpleImputer
        import numpy as np
        s2 = SimpleImputer(
            missing_values=np.nan,  # los valores que faltan son del tipo ``np.nan`` (Pandas estándar)
            strategy='median',  # la estrategia elegida es cambiar el valor faltante por una constante
            #fill_value=0,  # la constante que se usará para completar los valores faltantes es un int64 = 0
            verbose=0,
            copy=True
        )
        data[numeric_columns] = s2.fit_transform(data[numeric_columns])
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
