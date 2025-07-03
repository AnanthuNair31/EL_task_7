import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV , cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np


df = pd.read_csv("breast-cancer.csv")

print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())

df_clean = df.drop(columns=['id'])
df_clean['diagnosis'] = df_clean['diagnosis'].map({'M':1, 'B':0})

x = df_clean.drop(columns=['diagnosis'])
y = df_clean['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler  = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.transform(x_test)

svm_linear = SVC(kernel = 'linear')
svm_linear.fit(x_train_sc, y_train)
y_pred_ln = svm_linear.predict(x_test_sc)
linear_acc = accuracy_score(y_test, y_pred_ln)

svm_rbf = SVC(kernel = 'rbf')
svm_rbf.fit(x_train_sc, y_train)
y_pred_rbf = svm_rbf.predict(x_test_sc)
rbf_acc = accuracy_score(y_test, y_pred_rbf)

print(linear_acc, rbf_acc)


x_vis = x[['radius_mean', 'texture_mean']]
x_vis_train , x_vis_test , y_vis_train , y_vis_test = train_test_split(x_vis, y, test_size = 0.2, random_state = 42)

scaler_vis = StandardScaler()
x_vis_train_sc = scaler_vis.fit_transform(x_vis_train)
x_vis_test_sc = scaler_vis.transform(x_vis_test)

svm_ln_2d = SVC(kernel = 'linear')
svm_ln_2d.fit(x_vis_train_sc, y_vis_train)

svm_rbf_2d = SVC(kernel = 'rbf')
svm_rbf_2d.fit(x_vis_train_sc, y_vis_train)

h= 0.02
X_min, X_max = x_vis_train_sc[:, 0].min() - 1, x_vis_train_sc[:, 0].max() + 1
Y_min, Y_max = x_vis_train_sc[:, 1].min() - 1,x_vis_train_sc[:, 1].max() + 1
XX,YY = np.meshgrid(np.arange(X_min, X_max, h), np.arange(Y_min, Y_max, h))

Z_linear = svm_ln_2d.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
Z_rbf = svm_rbf_2d.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

fig, axs = plt.subplots(1, 2, figsize = (12,6))

axs[0].contour(XX, YY, Z_linear, cmap=plt.cm.coolwarm, alpha=0.8)
axs[0].scatter(x_vis_train_sc[:, 0], x_vis_train_sc[:, 1], c=y_vis_train,cmap=plt.cm.coolwarm, edgecolors='k')
axs[0].set_title("Linear SVM")
axs[0].set_xlabel(x_vis.columns[0])
axs[0].set_ylabel(x_vis.columns[1])

axs[1].contour(XX, YY, Z_rbf, cmap=plt.cm.coolwarm, alpha=0.8)
axs[1].scatter(x_vis_train_sc[:, 0], x_vis_train_sc[:, 1], c=y_vis_train ,cmap=plt.cm.coolwarm, edgecolors='k')
axs[1].set_title("RBF SVM")
axs[1].set_xlabel(x_vis.columns[0])
axs[1].set_ylabel(x_vis.columns[1])
plt.tight_layout()
plt.show()


parameter_grid = {'C': [0.1 , 1 ,  10 , 100],
                  'gamma':[0.001 , 0.01 , 0.1 , 1],
                  'kernel': ['rbf']
                  }

svm = SVC()

grid_search = GridSearchCV(estimator=svm, param_grid=parameter_grid,  scoring='accuracy', cv=5, verbose=1)
grid_search.fit(x_train_sc, y_train)

print("Best parameters  values:", grid_search.best_params_)
print("Best cross validation:", grid_search.best_score_)


best_svm = SVC(kernel='rbf', C=10, gamma=0.01)
cv_scores = cross_val_score(best_svm, x_train_sc, y_train, cv=5 , scoring='accuracy')

print("Cross-validation scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())