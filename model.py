import numpy as np
import pandas as pd
import time
import csv
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#Rough fitting a RandomForest to determine feature importance...
def rfForFeature(X,y,sample_weight):
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
    print "\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance)...\n"#, \
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

#Random forest for best parameters
def RFprediction(train_X,train_y,y_weights,test_X):
    sqrtfeat = int(np.sqrt(train_X.shape[1]))
    #print "sqrtfeat:", sqrtfeat
    minsampsplit = int(train_X.shape[0]*0.015)
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

    best_parameters = {"n_estimators": 10000,\
                       "max_features": sqrtfeat,\
                       "min_samples_split": minsampsplit}
    #================================================================
    #best_parameters = dict();  
    #best_parameters = clf.best_estimator_.get_params()  
    #for param_name in sorted(best_parameters.keys()):  
    #    print("\t%s: %r" % (param_name, best_parameters[param_name]));
    #================================================================

    print("Generating RandomForestClassifier model with parameters: ")
    forest = RandomForestClassifier(n_jobs=-1, oob_score=True, **best_parameters)
    forest.fit(train_X,train_y)
    #a nparray result
    result=forest.predict(test_X).astype(int)
    return result

#Logostic regression for best parameters
def LRprediction(train_X,train_y,y_weights,test_X):
    tuned_parameters =[{'penalty': ['l1'], 'tol': [1e-3, 1e-4],\
                     'C': [1, 10, 100, 1000]},\
                    {'penalty': ['l2'], 'tol':[1e-3, 1e-4],\
                     'C': [1, 10, 100, 1000]}]
    clf =GridSearchCV(LogisticRegression(), tuned_parameters, cv=5,\
                      fit_params={ 'sample_weight': y_weights})
    clf.fit(train_X,train_y)
    print("Best score: %0.3f" % clf.best_score_)
    best_parameters = dict()
    best_parameters = clf.best_estimator_.get_params()  
    for param_name in sorted(best_parameters.keys()):  
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    

    print("Generating LogisticRegressionClassifier model with parameters: ")
    lgr = LogisticRegression(**best_parameters)
    lgr.fit(train_X,train_y)
    #a nparray result
    result=lgr.predict(test_X).astype(int)
    return result

#do pca,return the new datagram train_x
def PCA_me(train_X,train_y):
    variance_pct = .99
    pca = PCA(n_components=variance_pct)    
    # Transform the initial features
    X_transformed = pca.fit_transform(X,y)    
    # Create a data frame from the PCA'd data
    pcaDataFrame = pd.DataFrame(X_transformed)
    return pcaDataFrame














    
