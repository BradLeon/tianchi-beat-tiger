import numpy as np
import pandas as pd
import time
import csv
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#使用随机森林计算特征重要性，进行特征选择
def rfForFeature(X,y,y_weights,features_list):
    forest = RandomForestClassifier(oob_score=True, n_estimators=10000)
    forest.fit(X, y, sample_weight=y_weights) 
    feature_importance = forest.feature_importances_
    
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    #print "Feature importances:\n", feature_importance
    
    fi_threshold = 18
    
    important_idx = np.where(feature_importance > fi_threshold)[0]
    #print "Indices of most important features:\n", important_idx
    
    important_features = features_list[important_idx]
    print("\n"+important_features.shape[0]+"Important features(>"+ fi_threshold+ "% of max importance)...\n")#, \
            #important_features
    
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    
    # Plot feature importance
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.draw()
    plt.show()

#随机森林算法做预测
def RFprediction(train_X,train_y,y_weights,test_X):
    sqrtfeat = int(np.sqrt(train_X.shape[1]))
    #print "sqrtfeat:", sqrtfeat
    minsampsplit = int(train_X.shape[0]*0.015)
    #使用pipeline构造参数范围，用GridSearch选取最优参数
    '''
    tuned_parameters = { "n_estimators": [5000, 10000],\
                "max_features": np.rint(np.linspace(sqrtfeat, sqrtfeat+2,3))\
                .astype(int),
                "min_samples_split":np.rint(np.linspace(train_X.shape[0]*.01,\
                                    train_X.shape[0]*.05, 3)).astype(int) }
    forest = RandomForestClassifier(oob_score=True, n_estimators=10000)
    clf =GridSearchCV(forest, tuned_parameters, cv=2 \
                      #,fit_params={ 'sample_weight': y_weights}\
                      #,scoring=['precision','recall']
                      )
    clf.fit(train_X,train_y)
    print("Best score: %0.3f" % clf.best_score_)
    '''
    #================================================================
    #best_parameters = dict();  
    #best_parameters = clf.best_estimator_.get_params()  
    #for param_name in sorted(best_parameters.keys()):  
    #    print("\t%s: %r" % (param_name, best_parameters[param_name]));
    #================================================================
    #经验参数
    best_parameters = {"n_estimators": 10000,\
                       "max_features": sqrtfeat,\
                       "min_samples_split": minsampsplit}

    print("Generating RandomForestClassifier model with parameters: ")
    forest = RandomForestClassifier(n_jobs=-1, oob_score=True, **best_parameters)
    forest.fit(train_X,train_y)
    #a nparray result
    result=forest.predict(test_X).astype(int)
    return result

#使用逻辑回归进行预测
def LRprediction(train_X,train_y,test_X):
    #使用pipeline构造参数范围，用GridSearch选取最优参数，一般来讲penalty应该是l2
    '''
    tuned_parameters =[{'penalty': ['l1'], 'tol': [1e-3, 1e-4],\
                     'C': [1, 10, 100, 1000], 'class_weight':'auto'},\
                    {'penalty': ['l2'], 'tol':[1e-3, 1e-4],\
                     'C': [1, 10, 100, 1000],'class_weight':'auto'}]
    '''
    tuned_parameters =[{'penalty': ['l2'], 'tol': [1e-3, 1e-4],\
                     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'class_weight':['auto']}]
    #求最佳参数
    clf =GridSearchCV(LogisticRegression(), tuned_parameters, cv=5, loss_func='log', score_func='f1')
    clf.fit(train_X,train_y)
    print("Best score: %0.3f" % clf.best_score_)
    #获得最佳参数
    best_parameters = clf.best_estimator_.get_params()  
    for param_name in sorted(best_parameters.keys()):  
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    

    print("Generating LogisticRegressionClassifier model with parameters: ")
    #使用最佳参数做驯良
    lgr = LogisticRegression(**best_parameters)
    lgr.fit(train_X,train_y)
    #a nparray result
    #预测
    result=lgr.predict(test_X)
    return result

#逻辑回归简单的测试函数，使用经验参数做训练
def LRprediction2(train_X,train_y,test_X):
    tuned_parameters ={'penalty': 'l2', 'tol': 1e-4,\
                     'C': 1,'class_weight' :{ 0: 2.0 , 1 : 1.0}}
    lgr = LogisticRegression(**tuned_parameters)
    lgr.fit(train_X,train_y)
    result=lgr.predict(test_X)
    return result

#PCA算法做维规约
def PCA_me(train_X,train_y):
    variance_pct = .99
    pca = PCA(n_components=variance_pct)    
    # Transform the initial features
    X_transformed = pca.fit_transform(train_X,train_y)    
    # Create a data frame from the PCA'd data
    pcaDataFrame = pd.DataFrame(X_transformed)
    return pcaDataFrame
#处理预测得到的结果，取预测结果为1的item_id和所给商品集的交集，产生最终的提交结果
def proResult(filepath,resultFilePath):
    origin=pd.read_csv(filepath,header=0)
    #读入商品子集表
    item=pd.read_csv('E:\\ZP\\tianchi\\tianchi_mobile_recommend_train_item.csv',header=0)
    item.drop(['item_geohash','item_category'],axis =1,inplace = True)
    item.drop_duplicates(inplace=True)
    
    origin = origin[origin.buy==1]
    origin.drop(['buy'],axis=1,inplace=True)
    #训练的结果和商品子集取交集
    origin = pd.merge(left=origin,right=item,how='inner',on='item_id')
    origin.drop_duplicates(['user_id','item_id'],inplace=True)
    
    #写入文件
    predictions_file = open(resultFilePath, "w",newline='')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['user_id','item_id'])
    open_file_object.writerows(origin.values)
    predictions_file.close()

if __name__ == '__main__':
    train=pd.read_csv('train_downupSample.csv',header=0)
    test=pd.read_csv('test.csv',header=0)
    
    #前两列为user_id item_id，不加入训练
    train_X = train.values[:,2:-1]
    #训练集的target列
    train_y = train.values[:,-1]
    test_X = test.values[:,2:]
    result = LRprediction2(train_X,train_y,test_X)
    print(type(result))
    print(result)
    
    submission = pd.DataFrame({'user_id':test.user_id,'item_id':test.item_id,'buy':result})
    submission.to_csv("submission.csv",index=False)
    
    proResult("submission.csv","submission_delete.csv")
    #submission = np.asarray(zip(test['user_id'].values, test['item_id'].values,result))
    #np.savetxt("submission.csv", submission, delimiter=",")
    
    '''
    predictions_file = open("result.csv", "w",newline='')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['user_id','item_id','buy'])
    open_file_object.writerows(list(submission))
    predictions_file.close()
    '''












