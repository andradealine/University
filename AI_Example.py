'''
This is an example of sklearn being used for Machine Learning. The objective is to predict if 
a person have dyslexia using well known supervisioned machine learning algorithm.
The data is not publicly available due to patience protection.
'''
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics
from sklearn import svm
import numpy as np
import pandas as pd
from math import floor, sqrt
import time
import matplotlib.pyplot as plt


classifiers = {
  "forest": RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None, min_samples_split=2,
               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
               min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=6, random_state=None, verbose=0,
               warm_start=False, class_weight=None),
 "knn": KNeighborsClassifier(n_neighbors=11, weights='distance',
          algorithm='ball_tree', leaf_size=50, p=2, metric='minkowski',
          metric_params=None, n_jobs=1),
 "svm": svm.SVC(kernel="linear", class_weight="balanced"),
 "adaboost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=True),
              n_estimators = 100, learning_rate=0.01, algorithm='SAMME.R', random_state=None),
 "bagging": BaggingClassifier(DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None,
              min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
              random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight='balanced', presort=False),
              n_estimators=100, max_samples=1.0, max_features=1.0,bootstrap=True, bootstrap_features=False, oob_score=False,
              warm_start=False, n_jobs=1, random_state=None, verbose=0),
 "mpl": MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto',
         beta_1=0.9, beta_2=0.999, early_stopping=False,
         epsilon=1e-08, learning_rate='invscaling',
         learning_rate_init=0.3, max_iter=500, momentum=0.2,
         nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
         solver='lbfgs', tol=0.00001, validation_fraction=0.1, verbose=False,
         warm_start=False)
}

testar_folds = [
  "forest",
  "knn",
  "svm",
  "adaboost",
  "bagging",
  "mpl"
]

testar_loo = [
  "forest",
  "knn",
  # "svm",
  # "adaboost",
  "bagging",
  # "mpl"
]

normalizar = [
  # "forest",
  "knn",
  "svm",
  # "adaboost",
  # "bagging",
   "mpl"
]

num_folds = 5
vezes = 2

tempo_treinos_loo = {clas: [] for clas in testar_loo}
tempo_uso_loo = {clas: [] for clas in testar_loo}
matrizes_loo = {clas: np.zeros((2, 2)) for clas in testar_loo}
accs_loo_bal = {clas: [0.0] for clas in testar_loo}
accs_loo = {clas: [0.0] for clas in testar_loo}

tempo_treinos_folds = {clas: [] for clas in testar_folds}
tempo_uso_folds = {clas: [] for clas in testar_folds}
matrizes_folds = {clas: [np.zeros((2, 2)) for t in range(vezes)] for clas in testar_folds}
accs_folds_bal = {clas: [0.0 for t in range(vezes)] for clas in testar_folds}
accs_folds = {clas: [0.0 for t in range(vezes)] for clas in testar_folds}

data = pd.read_csv('arquivo_DDA_45.csv')
data = data[['duration', 'frames', 'norm_pos_x', 'norm_pos_y', 'dispersion', 'avg_pupil_size', 'nome']]
data = data[data['dispersion'] < 2.0]
data = data[data['duration'] < 5.0]

data.replace("DDA", 0, inplace=True)
data.replace("N", 1, inplace=True)
data_dda = data[data['nome'] == 0]
data_n = data[data['nome'] == 1]

for clas in testar_loo:
  loo = LeaveOneOut().split(data.iloc[:,:-2], data.iloc[:,-1])
  for train_index, test_index in loo:
    clf = classifiers[clas]
    print("Class: {1}, Test: {0}".format(test_index, clas))
    if clas in normalizar:
      scaler =  MinMaxScaler().fit(data.iloc[train_index, :-2])
      X_train = scaler.transform(data.iloc[train_index, :-2])
      Y_train = data.iloc[train_index, -1]
      X_test = scaler.transform(data.iloc[test_index, :-2])
      Y_test = data.iloc[test_index, -1]
    else:
      X_train = data.iloc[train_index, :-2]
      Y_train = data.iloc[train_index, -1]
      X_test = data.iloc[test_index, :-2]
      Y_test = data.iloc[test_index, -1]
    iniciot = time.time()
    clf = clf.fit(X_train, Y_train)
    fimt = time.time()
    tempo_treinos_loo[clas].append(fimt-iniciot)
    inicio = time.time()
    pred = clf.predict(X_test)
    fim = time.time()
    tempo_uso_loo[clas].append(fim-inicio)
    matrizes_loo[clas] += (sklearn.metrics.confusion_matrix(Y_test, pred))
  m_loo = matrizes_loo[clas]
  accs_loo_bal[clas][0] += (0.5*(m_loo[0,0]/(m_loo[0,0]+m_loo[0,1]) + m_loo[1,1]/(m_loo[1,0]+m_loo[1,1])))
  accs_loo[clas][0] += (m_loo[0,0] + m_loo[1,1])/(m_loo[0,0] + m_loo[0,1] + m_loo[1,1] + m_loo[1,0])

for t in range(vezes):
  for clas in testar_folds:
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True).split(data.iloc[:,:-2], data.iloc[:,-1])
    print("Class: {1}, Test: {0}".format(t, clas))
    for train_index, test_index in folds:
      clf = classifiers[clas]
      if clas in normalizar:
        scaler =  MinMaxScaler().fit(data.iloc[train_index, :-2])
        X_train = scaler.transform(data.iloc[train_index, :-2])
        Y_train = data.iloc[train_index, -1]
        X_test = scaler.transform(data.iloc[test_index, :-2])
        Y_test = data.iloc[test_index, -1]
      else:
        X_train = data.iloc[train_index, :-2]
        Y_train = data.iloc[train_index, -1]
        X_test = data.iloc[test_index, :-2]
        Y_test = data.iloc[test_index, -1]
      iniciot = time.time()
      clf = clf.fit(X_train, Y_train)
      #print(clf.get_params())
      fimt = time.time()
      tempo_treinos_folds[clas].append(fimt-iniciot)
      inicio = time.time()
      pred = clf.predict(X_test)
      fim = time.time()
      tempo_uso_folds[clas].append(fim-inicio)
      matrizes_folds[clas][t] += (sklearn.metrics.confusion_matrix(Y_test, pred))


    m_folds = matrizes_folds[clas][t]
    accs_folds_bal[clas][t] += (0.5*(m_folds[0,0]/(m_folds[0,0]+m_folds[0,1]) + m_folds[1,1]/(m_folds[1,0]+m_folds[1,1])))
    accs_folds[clas][t] += (m_folds[0,0] + m_folds[1,1])/(m_folds[0,0] + m_folds[0,1] + m_folds[1,1] + m_folds[1,0])

if len(testar_folds) > 0:
    dfaccs_folds = pd.DataFrame(accs_folds)
    dfaccs_folds_bal = pd.DataFrame(accs_folds_bal)
    dffolds = pd.concat([dfaccs_folds, dfaccs_folds_bal], axis=1)
    dffolds.to_csv("acc_folds.csv")
    dfttreino_folds = pd.DataFrame(tempo_treinos_folds)
    dfttreino_folds.to_csv("ttreino_folds.csv")
    dftuso_folds = pd.DataFrame(tempo_uso_folds)
    dftuso_folds.to_csv("tuso_folds.csv")
    p = dfttreino_folds.plot.hist(alpha=0.4, bins=80)
    p.set_xlabel("Tempo [s]")
    p.set_ylabel("Frequencia")
    plt.savefig("plot_ttreino_folds.png")
    p = dftuso_folds.plot.hist(alpha=0.4, bins=80)
    p.set_xlabel("Tempo [s]")
    p.set_ylabel("Frequencia")
    plt.savefig("plot_tuso_folds.png")

if len(testar_loo) > 0:
    dfaccs_loo = pd.DataFrame(accs_loo)
    dfaccs_loo_bal = pd.DataFrame(accs_loo_bal)
    dfloo = pd.concat([dfaccs_loo, dfaccs_loo_bal], axis=1)
    dfloo.to_csv("acc_loo.csv")
    dfttreino_loo = pd.DataFrame(tempo_treinos_loo)
    dfttreino_loo.to_csv("ttreino_loo.csv")
    dftuso_loo = pd.DataFrame(tempo_uso_loo)
    dftuso_loo.to_csv("tuso_loo.csv")
    p = dfttreino_loo.plot.hist(alpha=0.4, bins=80)
    p.set_xlabel("Tempo [s]")
    p.set_ylabel("Frequencia")
    plt.savefig("plot_ttreino_loo.png")
    p = dftuso_loo.plot.hist(alpha=0.4, bins=80)
    p.set_xlabel("Tempo [s]")
    p.set_ylabel("Frequencia")
    plt.savefig("plot_tuso_loo.png")
