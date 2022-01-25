
# from sklearn.model_selection import train_test_split

# def train_test_split(
#     *arrays,              # 파이썬에서 *이 의미하는 것 : multi(여러 개 들어와도 된다. 한 개 이상의 의미!!)
#     test_size=None,       # test_size나 train_size 중 하나 적으면 된다 -> 합쳐서 1이 되도록(비율 써라) / default는 0.25, 0.75
#     train_size=None,      # 1보다 큰 수가 들어오면 데이터 건수로 취급
#     random_state=None,    # 씨드값 -> 랜덤으로 데이터를 섞는 방식을 정함(미리 어떻게 섞는지 정하는 난수 테이블이 만들어져 있고 거기서 어떻게 섞을 건지 정하는 것)
#                           # (고정해놓지 않으면 재실행 할 때마다 데이터셋과 트레인셋을 바꾸는 방식이 바뀐다.
#                           # 그러면 모델 재검정 할 때 실행 때마다 문제가 바뀌어서 가공을 잘 해서 점수가 좋아진 건지 문제가 바뀌어서 그런 건지 알 수가 없다.)
#                           # 컴퓨터마다 random seed 다르다. 가끔 seed 바꿔서 점수 올리려고 하는 경우도 있는데 다른 컴퓨터에서 점수 잘 나온 거 그대로 가져온다고 똑같은 점수 나오는 거 아님.
#                           # 딥러닝에서 작은 데이터로 하면 시드 따라 점수 확 달라지기도 ^^;;
#     shuffle=True,         # 섞어줘
#     stratify=None,        # 운 나쁘게 target데이터가 train과 test 셋에 편향되게 들어가면 점수 안 좋을 수도. -> 이걸 비율 일정하게 들어가게 정하는 게 stratify
# ):
# self 없으니 생성자 만들어서 ~~. 쓰는 거 아니다. 바로 train_test_split 쓴다.
# return list()로 나온다. -> listtype으로 준다는 게 아니라 여러 개를 준다.(list type이었으면 []로 썼을 것)
# 파이썬은 특이하게 return 여러 개 줄 수 있다.

# def add_mul(a, b):
#   return a+b, a*b
# res1, res2 = add_mul(4, 7)
# print(res1, res2) # 결과 11, 28


# df=pd.read_csv('./dataset/')
# train_test_split()

# trainX, testX = train_test_split(X) # X, y 따로따로 해도 된다.
# trainy, testy = train_test_split(y)
# trainX, testX, trainy, testy = train_test_split(X, y)


#------------------------------------
# # 학습하고 채점해보자
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# RandomForestClassifier는 Class
# accuracy_score : 함수

# rf=RandomForestClassifier(n_estimators=100)
# rf.fit(X_train, Y_train) #fit해라(학습해라) # sample_weight : 희박하게 나오는 거 맞추면 가중치 더 주겠다. / return: self(return 없음 메모리에만 저장한다.)
#                             # label encoder 이런 데서도 있었다.  /sklearn에서 fit은 return 없을 것임
# 내답안지=rf.predict(X_test) # 맞춰봐! (답지 안 줌) / X_test predict 해서 답안지에 넣어라
# rf.score() #class ClassifierMixin: 있는 거 흔히들 쓰긴 한데 더 정확하고 자주 쓰이는 거 쓰자. -> 사이킷런에서 metrics and scoring 파트
# # 3.3. Metrics and scoring: quantifying the quality of predictions
# # https://scikit-learn.org/stable/modules/model_evaluation.html
# # calssification -> accuracy() 쓰자. 왜? 타이타닉 evaluation  가면 accuracy로 채점한댔거든.
#
# acc_score=accuracy_score(y_test, 내답안지)
# #self 없다 -> ~. 쓰는 거 아님!!
# #return : float으로 된 점수 주겠다.
#
# print(f"정확도 : {acc_score:4f}") # 일반적으로 점수는 소수점 네자리까지 봐야된다. / 소수점 네 자리까지 가져오기
                                 # 지금은 단순한 거라 하난데 나중에 되면 점수 많아져서 format형 쓰는 게 편하다.

##########################

# from sklearn.linear_model import LogisticRegression # 얘도 알고리즘 모델
#                                                     # linear에 있지만 분류!!!!
#                                                     # 설명 주피터에 썼다.
#
#
# lg=LogisticRegression()
#
# lg.__class__.__name__


import numpy as np
import pandas as pd

s2 = pd.Series(['a',np.nan]) ## 이 경우 1,2,3은 dtype object(글자)
print(s2.dtype)

pd.isnull

# 사이킷런에서 점수랑 관련된 건 다 metrics에 들어있다.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score #y_true, y_pred
from sklearn.metrics import precision_recall_curve # y_true, probas_pred
from sklearn.metrics import roc_auc_score, roc_curve # y_true, y_score
# 회귀는 모델이 분류는 메트릭스가 어렵다.
# 회귀는 코딩 지금까지 타이타닉 한 거에서 모델만 바꾸면 됨.

-------
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

1차 점수 보기
2차 gridCV 튜닝

pd.Series.to_dict()

pd.Series.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE

SMOTE

