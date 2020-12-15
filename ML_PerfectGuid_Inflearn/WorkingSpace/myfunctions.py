### 통상적인 encoding 방법
from sklearn import preprocessing
def encode_features(dataDF):
    features = dataDF.columns.tolist()      # 모든 컬럼을 불러온다
    for feature in features:                # 모든 컬럼이름에 대해 반복
        le = preprocessing.LabelEncoder()   # 엔코딩 객체 생성
        le = le.fit(dataDF[feature])        # 엔코딩 적합 - 모든 칼럼에 대해 각각 진행
        dataDF[feature] = le.transform(dataDF[feature]) # transform 진행 - 모든 칼럼에 대해 각각 진행
    return dataDF

### 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))

###
# 인자로 사이킷런의 Estimator객체와, 학습/테스트 데이터 세트를 입력 받아서 학습/예측/평가 수행.
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:, 1]
    get_clf_eval(tgt_test, pred, pred_proba)