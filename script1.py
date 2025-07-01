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


#-------/Random Forest/------------------------------------------------------------------------


rf_model = RandomForestRegressor()
rf_model.fit(X_bsq, y_bsq)
rf_importances = rf_model.feature_importances_


#-------/Regressões/---------------------------------------------------------------------------


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['bsq_score'], bins=20, kde=True)
plt.title('BSQ Score')

plt.subplot(1, 2, 2)
sns.histplot(df['eat_score'], bins=20, kde=True)
plt.title('EAT Score')
plt.tight_layout()
plt.savefig("histogramas.png")

plt.figure(figsize=(10, 5))
sns.boxplot(x='sexo', y='bsq_score', data=df)
plt.title('Boxplot BSQ por Sexo')
plt.savefig("boxplot_bsq_sexo.png")


corr_cols = ['idade', 'imc', 'freq_cod', 'bsq_score', 'eat_score']
plt.figure(figsize=(10, 8))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap - Correlações')
plt.savefig("heatmap_correlacao_com_freq.png")


#-------/Spearman Mann-Whitney/----------------------------------------------------------------


spearman_corr, spearman_p = spearmanr(df['bsq_score'], df['eat_score'])
bsq_masc = df[df['sexo'] == 'Masculino']['bsq_score']
bsq_fem = df[df['sexo'] == 'Feminino']['bsq_score']
mann_stat, mann_p = mannwhitneyu(bsq_masc, bsq_fem, alternative='two-sided')


#-------/Export/-------------------------------------------------------------------------------


with open("output_modelos.txt", "w") as f:
    f.write("=== Regressão Linear - BSQ ===\n")
    f.write(model_bsq.summary().as_text())
    f.write("\n\n=== Regressão Linear - EAT ===\n")
    f.write(model_eat.summary().as_text())
    f.write("\n\n=== Importância Random Forest (BSQ) ===\n")
    for var, imp in zip(X_bsq.columns, rf_importances):
        f.write(f"{var}: {imp:.4f}\n")
    f.write("\n\n=== Correlação de Spearman (BSQ vs EAT) ===\n")
    f.write(f"Spearman R = {spearman_corr:.4f}, p = {spearman_p:.4g}\n")
    f.write("\n\n=== Mann-Whitney (BSQ por Sexo) ===\n")
    f.write(f"U = {mann_stat:.4f}, p = {mann_p:.4g}\n")











