"""
This program runs Naive Bayes, SVM, KNN, PCA, an Ensemble-learning method combining
NB, SVM, and KNN, AdaBoost, and RandomForest on the data.

Authors: Emily Wu, Katrina Midgley
"""
from sklearn.naive_bayes import GaussianNB
from preProcessData import *
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import cross_validation
from scipy.stats import mode
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

def naiveBayes(x_train, x_test, y_train):
    """
    This function performs Naive Bayes prediciton on a given set of known data 
    split into test and training data.
    """
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    return y_pred

def SVM(x_train, x_test, y_train, kernel):
    """
    This function performs SVM prediction on a given set of known data split
    into test and training data. 
    """
    svc_model = SVC(kernel=kernel)
    svc_model.fit(x_train, y_train)
    y_pred = svc_model.predict(x_test)
    return y_pred

def KNN(x_train, x_test, y_train, k=3):
    """
    This function performs KNN prediction on a given set of known data split
    into test and training data. 
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    return y_pred

def ensemble(all_predict, size):
    """
    This function takes in a matrix of predicted classifications and the number of 
    rows and returns predictions based on the most common classification. It
    returns an array of predictions. 
    """
    mode_pred = np.zeros(shape=(size,1))
    for i in range(np.shape(all_predict)[1]):
        pred= mode(all_predict[:,i])
        # break ties randomly
        if pred[1] == 1:
            pred_val = random.randrange(2)
        else:
            pred_val = pred[0]
        mode_pred[i,0] = pred_val
    # return most common prediction
    return mode_pred

def main():
    points = "points"
    labels = "labels"
    atts = "att_names"
    water_data = Data(points, labels, atts, 0, 10000)
    
    # applying PCA
    n=7
    print "Number of principal components used: ", n
    pca = PCA(n_components = n)
    water_data.x_data = pca.fit_transform(water_data.x_data)
    
    test_size = 0.4
    
    all_predict = np.zeros(shape= (int(np.shape(water_data.x_data))[1]), int(np.shape(water_data.y_data)[0]*test_size)))

    # applying cross validation at 40% split
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(water_data.x_data, water_data.y_data, test_size = test_size)
    index = 0
    
    # Naive Bayes 
    bayes_pred = naiveBayes(x_train, x_test, y_train)
    total_bayes = (y_test != bayes_pred).sum()/float(np.shape(x_test)[0])
    all_predict[index,:] = bayes_pred.transpose()
    index += 1
    print "Bayes error rate: ", total_bayes
    
    # Support Vector Machine
    svm_pred = SVM(x_train, x_test, y_train, "rbf")
    total_svm = (y_test != svm_pred).sum()/float(np.shape(x_test)[0])
    all_predict[index,:] = svm_pred
    index +=1
    print "SVM error rate: ", total_svm
    
    # K nearest neighbors
    print "5"
    knn_pred = KNN(x_train, x_test, y_train, 5)
    total_knn = (y_test != knn_pred).sum()/float(np.shape(x_test)[0])
    all_predict[index,:] = knn_pred
    print "KNN error rate: ", total_knn

    # Support Vector Machine
    print "poly"
    svm_pred = SVM(x_train, x_test, y_train, "poly")
    total_svm = (y_test != svm_pred).sum()/float(np.shape(x_test)[0])
    all_predict[index,:] = svm_pred
    index +=1
    print "SVM error rate: ", total_svm
    
    # K nearest neighbors
    print "10"
    knn_pred = KNN(x_train, x_test, y_train, 10)
    total_knn = (y_test != knn_pred).sum()/float(np.shape(x_test)[0])
    all_predict[index,:] = knn_pred
    print "KNN error rate: ", total_knn

    # Support Vector Machine
    print "linear"
    svm_pred = SVM(x_train, x_test, y_train, "linear")
    total_svm = (y_test != svm_pred).sum()/float(np.shape(x_test)[0])
    all_predict[index,:] = svm_pred
    index +=1
    print "SVM error rate: ", total_svm
    
    # K nearest neighbors with k = 10
    print "10"
    knn_pred = KNN(x_train, x_test, y_train, 10)
    total_knn = (y_test != knn_pred).sum()/float(np.shape(x_test)[0])
    all_predict[index,:] = knn_pred
    print "KNN error rate: ", total_knn

    #K nearest neighbors with k = 20
    print "20"
    knn_pred = KNN(x_train, x_test, y_train, 20)
    total_knn = (y_test != knn_pred).sum()/float(np.shape(x_test)[0])
    all_predict[index,:] = knn_pred
    print "KNN error rate: ", total_knn  

    # Hand calculates ensemble error rate - predictions using average classification
    mode_pred = ensemble(all_predict, (int(np.shape(water_data.y_data)[0]*test_size)))
    y_test = np.asarray(y_test)
    y_test = y_test.reshape((-1,1))
    bool = np.asarray(y_test != mode_pred, dtype = np.bool)
    avg_error =  bool.sum()
    total_error = avg_error/float(np.shape(x_test)[0])
    print "Ensemble error rate", total_error
  
    # Adaboost
    ada = AdaBoostClassifier(n_estimators=100)
    scores = cross_validation.cross_val_score(ada,  water_data.x_data, water_data.y_data)
    #scores = cross_validation.cross_val_score(ada, x_train, y_train)
    print "Adaboost error: ",  1-scores.mean()

    # Random forest
    forest = RandomForestClassifier(n_estimators = 10)
    #forest.fit(x_train, y_train)
    #forest_pred = forest.predict(x_test)
    scores = cross_validation.cross_val_score(forest, water_data.x_data, water_data.y_data)
    #scores = cross_validation.cross_val_score(forest, x_train, y_train)
    print "Random forest error: ", 1-scores.mean()

if __name__ == '__main__':
    main()