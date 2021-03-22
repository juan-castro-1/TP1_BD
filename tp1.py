
"""
TRABAJO PRACTICO N°1 BIG DATA

JUAN CASTRO - SANTIAGO DE MARTINI - PABLO FERNANDEZ

"""


'PARTE 1'

'Ejercicio 2'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # Para matriz de correlaciones
import seaborn as sns            # Para grÃ¡ficos bonitos
import statsmodels.api as sm     # Para agregar la columna de 1 a la matriz X


os.getcwd()  
#os.chdir('C:\\Users\\juan_\\Desktop\\Big Data\\TP_1')
os.chdir('C:\\Users\\User\\Documents\\1. Maestria\\3. Trimestre III\\1. Big Data\\TP')

df = pd.read_excel(r'usu_individual_T120.xlsx', sheet_name='usu_pers_T12020')


'Eliminamos todos los aglomerados que no son BsAs'

'Apartado a'

df = df.set_index("AGLOMERADO")
df = df.drop([2,3,4,5,6,7,8,9,10,12,13,14,15,17,18,19,20,22,23,25,26,27,29,30,31,34,36,38,91,93], axis=0)

'Apartado b'

'Elimino los individuos con ed<0'
df = df[df['CH06']>=0]

'Elimino los individuos con un ingreso menor a cero P21 es el ingreso de la ocupacion principal'
df = df[df['P21']>=0]


'Apartado c'
'Grafico de barras de variables de genero con sus respectivos ponderadores'

import matplotlib.pyplot as pyplot

labels = ('Hombre', 'Mujer') 
index = (1, 2)
sizes = df['CH04'].value_counts()

pyplot.bar(index, sizes, tick_label=labels)
pyplot.ylabel('Cantidad de personas (en millones)', labelpad=13)
pyplot.xlabel('Sexo', labelpad=14)
plt.title("EPH Gran Buenos Aires - 1er trimestre 2020", y=1.02);
pyplot.show()


'Apartado d'

def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
        # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

df.columns
mycols = ['CH04','CH07','CH08','NIVEL_ED','ESTADO','CAT_INAC','IPCF']
corr = df[mycols].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)



'Apartado e'
    
df['ESTADO'].value_counts() 
means = df.groupby('ESTADO')['IPCF'].mean()
print(means)


'Apartado f' 

df2 = pd.read_excel(r'tabla_adulto_equiv.xlsx', sheet_name='Tabla de adulo equivalente')

df['adulto_equiv']="" 

df.loc[df.CH06 < 1, 'adulto_equiv'] = '0.35'
df.loc[df.CH06 == 1, 'adulto_equiv'] = '0.37'
df.loc[df.CH06 == 2, 'adulto_equiv'] = '0.46'
df.loc[df.CH06 == 3, 'adulto_equiv'] = '0.51'
df.loc[df.CH06 == 4, 'adulto_equiv'] = '0.55'
df.loc[df.CH06 == 5, 'adulto_equiv'] = '0.6'
df.loc[df.CH06 == 6, 'adulto_equiv'] = '0.64'
df.loc[df.CH06 == 7, 'adulto_equiv'] = '0.66'
df.loc[df.CH06 == 8, 'adulto_equiv'] = '0.68'
df.loc[df.CH06 == 9, 'adulto_equiv'] = '0.69'
df.loc[(df.CH06 == 10) & (df.CH04==1), 'adulto_equiv'] = '0.7'
df.loc[(df.CH06 == 10) & (df.CH04==2), 'adulto_equiv'] = '0.79'
df.loc[(df.CH06 == 11) & (df.CH04==1), 'adulto_equiv'] = '0.72'
df.loc[(df.CH06 == 11) & (df.CH04==2), 'adulto_equiv'] = '0.82'
df.loc[(df.CH06 == 12) & (df.CH04==1), 'adulto_equiv'] = '0.74'
df.loc[(df.CH06 == 12) & (df.CH04==2), 'adulto_equiv'] = '0.85'
df.loc[(df.CH06 == 13) & (df.CH04==1), 'adulto_equiv'] = '0.76'
df.loc[(df.CH06 == 13) & (df.CH04==2), 'adulto_equiv'] = '0.9'
df.loc[(df.CH06 == 14) & (df.CH04==1), 'adulto_equiv'] = '0.76'
df.loc[(df.CH06 == 14) & (df.CH04==2), 'adulto_equiv'] = '0.96'
df.loc[(df.CH06 == 15) & (df.CH04==1), 'adulto_equiv'] = '0.77'
df.loc[(df.CH06 == 15) & (df.CH04==2), 'adulto_equiv'] = '1'
df.loc[(df.CH06 == 16) & (df.CH04==1), 'adulto_equiv'] = '0.77'
df.loc[(df.CH06 == 16) & (df.CH04==2), 'adulto_equiv'] = '1.03'
df.loc[(df.CH06 == 17) & (df.CH04==1), 'adulto_equiv'] = '0.77'
df.loc[(df.CH06 == 17) & (df.CH04==2), 'adulto_equiv'] = '1.04'
df.loc[(df.CH06 >= 18) & (df.CH06 <= 29) & (df.CH04==1), 'adulto_equiv'] = '0.76'
df.loc[(df.CH06 >= 18) & (df.CH06 <= 29) & (df.CH04==2), 'adulto_equiv'] = '1.02'
df.loc[(df.CH06 >= 30) & (df.CH06 <= 45) & (df.CH04==1), 'adulto_equiv'] = '0.77'
df.loc[(df.CH06 >= 30) & (df.CH06 <= 45) & (df.CH04==2), 'adulto_equiv'] = '1'
df.loc[(df.CH06 >= 46) & (df.CH06 <= 60) & (df.CH04==1), 'adulto_equiv'] = '0.76'
df.loc[(df.CH06 >= 46) & (df.CH06 <= 60) & (df.CH04==2), 'adulto_equiv'] = '1'
df.loc[(df.CH06 >= 61) & (df.CH06 <= 75) & (df.CH04==1), 'adulto_equiv'] = '0.67'
df.loc[(df.CH06 >= 61) & (df.CH06 <= 75) & (df.CH04==2), 'adulto_equiv'] = '0.83'
df.loc[(df.CH06 >= 76) & (df.CH04==1), 'adulto_equiv'] = '0.63'
df.loc[(df.CH06 >= 76) &  (df.CH04==2), 'adulto_equiv'] = '0.74'


df['adulto_equiv'] = pd.to_numeric(df['adulto_equiv'])
df['ad_equiv_hogar'] = df['adulto_equiv'].groupby(df['CODUSU']).transform('sum')



'Ejercicio 3'
print((df['ITF'] == 0).value_counts()) 
respondieron   = df[df['ITF']>0]
norespondieron = df[df['ITF']==0]

'Ejercicio 4'
respondieron['ingreso_necesario']=""
respondieron['ingreso_necesario'] = 13286 * respondieron['ad_equiv_hogar'] 

norespondieron['ingreso_necesario']=""
norespondieron['ingreso_necesario'] = 13286 * norespondieron['ad_equiv_hogar'] 
'Ejercicio 5'
respondieron['pobre']=""
respondieron['pobre'] = np.where(respondieron['ITF'] >= respondieron['ingreso_necesario'], 0 , 1)
print((respondieron['pobre'] == 1).value_counts()) 


'PUNTO 2'

'Ejercicio 1'

respondieron = respondieron.drop(['PP08D1','PP08D4','PP08F1','PP08F2','PP08J1',
                                  'PP08J2','PP08J3','P21','DECOCUR','IDECOCUR',
                                  'RDECOCUR','GDECOCUR','PDECOCUR','ADECOCUR',
                                  'PONDIIO','TOT_P12','P47T','DECINDR','IDECINDR',
                                  'RDECINDR','GDECINDR','PDECINDR','ADECINDR',
                                  'PONDII','V2_M','V3_M','V4_M','V5_M','V8_M',
                                  'V9_M','V10_M','V11_M','V12_M','V18_M','V19_AM',
                                  'V21_M','T_VI','ITF','DECIFR','IDECIFR',
                                  'RDECIFR','GDECIFR','PDECIFR','ADECIFR',
                                  'IPCF','DECCFR','IDECCFR','RDECCFR',
                                   'GDECCFR','PDECCFR','ADECCFR','PONDIH',
                                   'adulto_equiv','ad_equiv_hogar','ingreso_necesario'], axis=1)

norespondieron = norespondieron.drop(['PP08D1','PP08D4','PP08F1','PP08F2','PP08J1',
                                  'PP08J2','PP08J3','P21','DECOCUR','IDECOCUR',
                                  'RDECOCUR','GDECOCUR','PDECOCUR','ADECOCUR',
                                  'PONDIIO','TOT_P12','P47T','DECINDR','IDECINDR',
                                  'RDECINDR','GDECINDR','PDECINDR','ADECINDR',
                                  'PONDII','V2_M','V3_M','V4_M','V5_M','V8_M',
                                  'V9_M','V10_M','V11_M','V12_M','V18_M','V19_AM',
                                  'V21_M','T_VI','ITF','DECIFR','IDECIFR',
                                  'RDECIFR','GDECIFR','PDECIFR','ADECIFR',
                                  'IPCF','DECCFR','IDECCFR','RDECCFR',
                                   'GDECCFR','PDECCFR','ADECCFR','PONDIH',
                                   'adulto_equiv','ad_equiv_hogar','ingreso_necesario'], axis=1)

  
'Ejercicio 2' 



train, test = train_test_split(respondieron,test_size=0.3, random_state=101)

train = train.set_index("CODUSU")
test = test.set_index("CODUSU")
'Ejercicio 3'

'Separamos la variable de interés y eliminamos fecha de nacimiento, '
'las variables constantes y aquellas que tienen missing values'

ytrain = train['pobre']
xtrain = train.drop(['pobre','CH05'], axis=1)
xtrain = xtrain.dropna(axis=1)

ytest = test['pobre']
xtest = test.drop(['pobre','CH05'], axis=1)
xtest = xtest.dropna(axis=1)


xtrain = xtrain[[c for c
        in list(xtrain)
        if len(xtrain[c].unique()) > 1]]

xtrain = sm.add_constant(xtrain, has_constant='add') 

xtest = xtest[[c for c
        in list(xtest)
        if len(xtest[c].unique()) > 1]]

xtest = sm.add_constant(xtest, has_constant='add') 


'Ejercicio 4'

'Logit'

logit_model = sm.Logit(ytrain.astype(float),xtrain.astype(float))
result=logit_model.fit()

print(result.summary2().as_latex())

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 

y_pred_log = result.predict(xtest)
y_pred_log = np.where(y_pred_log > 0.5 , 1 , y_pred_log)
y_pred_log = np.where(y_pred_log <=0.5 , 0 , y_pred_log)

from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(ytest, y_pred_log)
print(cm_log)   
print('Accuracy Score :',accuracy_score(ytest, y_pred_log)) 



from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()  

fpr, tpr, thresholds = roc_curve(ytest, y_pred_log)
plot_roc_curve(fpr, tpr)

auc_log = roc_auc_score(ytest, y_pred_log)
print('AUC: %.2f' % auc_log)

'Linear Discriminant Analysis'

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(xtrain, ytrain)
resultslda=lda.predict(xtest)
y_pred_lda=pd.Series(resultslda.tolist())


cm_lda = confusion_matrix(ytest, y_pred_lda)
print(cm_lda)   

auc_lda = roc_auc_score(ytest, y_pred_lda)
print('AUC: %.2f' % auc_lda)
fpr, tpr, thresholds = roc_curve(ytest, y_pred_lda)
plot_roc_curve(fpr, tpr)

print('Accuracy Score :',accuracy_score(ytest, y_pred_lda)) 

'KNN, k=3'

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_est = knn.fit(xtrain, ytrain) 
y_pred_knn = knn.predict(xtest)
confusion_matrix(ytest, y_pred_knn) 
auc_knn = roc_auc_score(ytest, y_pred_knn)

print('AUC: %.2f' % auc_knn)
fpr, tpr, thresholds = roc_curve(ytest, y_pred_knn)
plot_roc_curve(fpr, tpr)

print('Accuracy Score :',accuracy_score(ytest, y_pred_knn)) 


'Ejercicio 5'

# MSE

import statistics 

mse_log = statistics.mean((y_pred_log - ytest)**2)
mse_log

y_pred_lda = pd.Series(y_pred_lda).array
mse_lda = statistics.mean((y_pred_lda - ytest)**2)
mse_lda

mse_knn = statistics.mean((y_pred_knn - ytest)**2)
mse_knn

print("MSE")
print("logit", mse_log)
print("lda",  mse_lda)
print("knn",  mse_knn)


'Ejercicio 6'

my_list = list(xtrain)
print(my_list)
norespondieron = sm.add_constant(norespondieron, has_constant='add') 
norespon = norespondieron[my_list]

y_pred_knn_new = knn.predict(norespon)

num_pobre = np.count_nonzero(y_pred_knn_new)
num_rows = y_pred_knn_new.shape
num_pobre



'Ejercicio 7'

'Seleccionamos solo las variables que tienen baja multicolinealidad así se reduce la varianza de los estimadores'


respondieron_2 = respondieron.filter(['COMPONENTE', 'H15', 'CH03', 'CH04', 'CH06', 'CH08','CH11',
                                      'CH12','CH13','NIVEL_ED','CAT_INAC','PP02C6','pobre','CODUSU'], axis=1)
train_2, test_2 = train_test_split(respondieron_2,test_size=0.3, random_state=101)


train_2 = train_2.set_index("CODUSU")
test_2 = test_2.set_index("CODUSU")

ytrain_2 = train_2['pobre']
xtrain_2 = train_2.drop(['pobre'], axis=1)


ytest_2 = test_2['pobre']
xtest_2 = test_2.drop(['pobre'], axis=1)

xtrain_2 = sm.add_constant(xtrain_2, has_constant='add') 


xtest_2 = sm.add_constant(xtest_2, has_constant='add') 

'Logit'

logit_model_2 = sm.Logit(ytrain_2.astype(float),xtrain_2.astype(float))
result_2=logit_model_2.fit()

print(result_2.summary2().as_latex())

y_pred_log_2 = result_2.predict(xtest_2)
y_pred_log_2 = np.where(y_pred_log_2 > 0.5 , 1 , y_pred_log_2)
y_pred_log_2 = np.where(y_pred_log_2 <=0.5 , 0 , y_pred_log_2)

cm_log_2 = confusion_matrix(ytest_2, y_pred_log_2)
print(cm_log_2)   

auc_log_2 = roc_auc_score(ytest_2, y_pred_log_2)
print('AUC: %.2f' % auc_log_2)

fpr, tpr, thresholds = roc_curve(ytest_2, y_pred_log_2)
plot_roc_curve(fpr, tpr)


import statistics 

mse_log_2 = statistics.mean((y_pred_log_2 - ytest_2)**2)
mse_log_2



