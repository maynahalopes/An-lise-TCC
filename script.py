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
    'freq_treino': np.random.choice(['1x', '2-3x', 'Todos os dias'], 100)
})


for col in bsq_cols:
    df[col] = np.random.randint(1, 7, size=100)
for col in eat_cols:
    df[col] = np.random.randint(1, 7, size=100)






#==============================================================================================
