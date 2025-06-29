import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr, mannwhitneyu, pearsonr


#==============================================================================================


np.random.seed(42)
bsq_cols = [f"BSQ_{i}" for i in range(1, 35)]
eat_cols = [f"EAT_{i}" for i in range(1, 27)]

df = pd.DataFrame({
    'idade': np.random.randint(18, 40, 100),
    'sexo': np.random.choice(['Masculino', 'Feminino'], 100),
    'peso': np.random.randint(50, 100, 100),
    'altura': np.random.randint(150, 190, 100),
    'freq_treino': np.random.choice(['1x', '2-3x', 'Todos os dias'], 100),
    'bsq_score': np.random.randint(34, 204, 100),
    'eat_score': np.random.randint(26, 156, 100)
})


for col in bsq_cols:
    df[col] = np.random.randint(1, 7, size=100)
for col in eat_cols:
    df[col] = np.random.randint(1, 7, size=100)



df['imc'] = df['peso'] / (df['altura'] / 100)**2
df['bsq_score'] = df[bsq_cols].sum(axis=1)
df['eat_score'] = df[eat_cols].sum(axis=1)

le_sexo = LabelEncoder()
df['sexo_cod'] = le_sexo.fit_transform(df['sexo'])
le_freq = LabelEncoder()
df['freq_cod'] = le_freq.fit_transform(df['freq_treino'])











#-------/Regressões/---------------------------------------------------------------------------


X_bsq = sm.add_constant(df[['idade', 'sexo_cod', 'imc', 'freq_cod']])
y_bsq = df['bsq_score']
model_bsq = sm.OLS(y_bsq, X_bsq).fit()

X_eat = sm.add_constant(df[['idade', 'sexo_cod', 'imc', 'freq_cod']])
y_eat = df['eat_score']
model_eat = sm.OLS(y_eat, X_eat).fit()
