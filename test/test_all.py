from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier ,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# 根据建议，这里不需要使用随机数 https://stackoverflow.com/questions/42191717/scikit-learn-random-state-in-splitting-dataset
# np.random.seed()
RANDOM_SEED =  0

# 整理数据， 输出ACC，RE，Pr，F1， MCC5个评估值
def metri(true_labels_cv,predictions,m):
    ACC = []
    Re = []
    Pr = []
    F1= []
    MCC = []
    AUC = []
    ALL = []
    print(confusion_matrix(true_labels_cv, predictions))
    #print ('ACC: %.4f' % metrics.accuracy_score(true_labels_cv, predictions))
    #print ('Recall: %.4f' % metrics.recall_score(true_labels_cv, predictions,pos_label=1))
    #print ('Precesion: %.4f' %metrics.precision_score(true_labels_cv, predictions))
    #print ('F1-score: %.4f' %metrics.f1_score(true_labels_cv, predictions))
    rig=[1]
    test_y_1=[+1 if x in rig else -1 for x in true_labels_cv]
    y_pred_1=[+1 if x in rig else -1 for x in predictions]
    #print ('MCC: %.4f' %metrics.matthews_corrcoef(test_y_1,y_pred_1))
    #print ('AUC: %.4f' % metrics.roc_auc_score(true_labels_cv, predictions))
    ACC.append(metrics.accuracy_score(true_labels_cv, predictions))
    Re.append(metrics.recall_score(true_labels_cv, predictions,pos_label=1))
    Pr.append(metrics.precision_score(true_labels_cv, predictions))
    F1.append(metrics.f1_score(true_labels_cv, predictions))
    MCC.append(metrics.matthews_corrcoef(test_y_1,y_pred_1))
    #AUC.append(metrics.roc_auc_score(true_labels_cv, predictions))
    #ALL = ACC+Re+Pr+F1+MCC+AUC
    if m==1:
        return ACC
    if m==2:
        return Re
    if m==3:
        return Pr
    if m==4:
        return F1
    if m==5:
        return MCC
    else:
        ALL = ACC+Re+Pr+F1+MCC
        return ALL

#####输入已经过特征选择后的特征向量集

X_train= pd.read_csv('1471train.csv')
X_test= pd.read_csv('159test.csv')

#特征名称
featuresname = list(X_train.columns[1:])  #la colonne 0 est le quote_conversionflag  
print(featuresname)

X_train=X_train[featuresname]
X_test=X_test[featuresname]

print(X_train)
print(X_test)

#引入训练集类别
df = pd.read_csv('1471labels.csv',header=None)
y_train = []
y_train=df.iloc[:,0]
#引入测试集类别
df1 = pd.read_csv('159labels.csv',header=None)
y_test = []
y_test=df1.iloc[:,0]

# 定义 svm模型使用的参数
svm_alg = SVC(gamma=0.03125, C = 4, random_state = RANDOM_SEED)

# 对训练集和测试集进行min max预处理
min_max = preprocessing.MinMaxScaler()
min_max.fit(X_train)
trainplus = min_max.transform(X_train)
testplus = min_max.transform(X_test)

# 调用SVM算法，并输出结果
model=svm_alg.fit(trainplus, y_train)
y_pred = model.predict(testplus)
#print(y_pred)
ALL9 = metri(y_test,y_pred,6)
#result.append(ALL9)    
print("打印SVM计算结果")
print(ALL9)
