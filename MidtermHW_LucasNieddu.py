#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:42:38 2020

@author: Lucas Nieddu 
@assignment: Midterm hw - Machine Learning
"""
#Basic imports  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Sklearn imports 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

# Model Imports from sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm as svm 



class MidTermHW():

    # splits the data
    def splitData(data):
        try:
            train_data, test_data = train_test_split(
                data,test_size=0.3,train_size=0.7)
            print("Data successfully split\n")
            #drop target column from train_data
            y_train = train_data.iloc[:,0]
            x_train = train_data.drop("label", axis = 1)
            y_train = y_train.to_numpy()
            x_train = x_train.to_numpy()
            x_test = test_data.iloc[:,0]
            y_test = test_data.drop("label", axis = 1)
        except Exception as e:
            print("Error occured{}".format(e))
        return x_train, y_train, y_test, x_test
    
    # plots the digits 
    def plot_digits(instances, images_per_row=10, **options):
        size = 28
        images_per_row = min(len(instances), images_per_row)
        images = [instance.reshape(size,size) for instance in instances]
        n_rows = (len(instances) - 1) // images_per_row + 1
        row_images = []
        n_empty = n_rows * images_per_row - len(instances)
        images.append(np.zeros((size, size * n_empty)))
        for row in range(n_rows):
            rimages = images[row * images_per_row : (row + 1) * images_per_row]
            row_images.append(np.concatenate(rimages, axis=1))
            image = np.concatenate(row_images, axis=0)
            plt.imshow(image, cmap = mpl.cm.binary, **options)
            plt.axis("off")
    
    def plot_roc_curve(fpr, tpr, label = None):
            plt.plot(fpr, tpr, linewidth=2, label=label)
            plt.plot([0,1], [0,1], 'k--') # dashboard diagonal
            plt.show()
  
    
    #Stochastic Gradient Descent
    def evaluateBinarySGD(x_train, y_train_5, five):
        sgd_clf_5 = SGDClassifier(random_state = 42)
        sgd_clf_5.fit(x_train, y_train_5)
        sgd_pred_5 = sgd_clf_5.predict([five])
        print("SSGD Model prediction for image:\n", sgd_pred_5, "\n")
        
        #Accuracy rating
        sgd_accuracy = cross_val_score(sgd_clf_5, x_train,
                                       y_train_5, cv=3, scoring = "accuracy")
        print("SGD Accuracy:\n", sgd_accuracy, "\n")
        # Confusion Matrix
        # Much better way to evaluate performance of a classifier
        y_train_5_pred = cross_val_predict(sgd_clf_5, x_train, y_train_5, cv=3)
        sgd_conf_matrix = confusion_matrix(y_train_5, y_train_5_pred)
        print("SGD Confusion matrix:\n", sgd_conf_matrix, "\n")
        # How to decide which threshold to use
        sgd_y_scores = cross_val_predict(sgd_clf_5, x_train, y_train_5, cv=3, 
                             method="decision_function")
        sgd_precisions, sgd_recalls, sgd_thresholds = precision_recall_curve(
                y_train_5, sgd_y_scores)
        # plot precision vs recall
        plt.title("SGD Precision vs Recall")
        plt.plot(sgd_recalls, sgd_precisions, "b-", linewidth=2)
        plt.xlabel("Recall", fontsize=16)
        plt.ylabel("Precision", fontsize=16)
        plt.axis([0, 1, 0, 1])
        plt.figure(figsize=(8, 6))
        plt.show()
        
        # Probably want to select a precision/recall trade-off 
        #just before the drop--> around 60%
        threshold_90_precision = sgd_thresholds[np.argmax(sgd_precisions>=0.60)]
        y_train_pred_90 = (sgd_y_scores >= threshold_90_precision)
        sgd_precision = precision_score(y_train_5, y_train_pred_90)
        print("SGD precision score:\n", sgd_precision, "\n")
        sgd_recall = recall_score(y_train_5, y_train_pred_90)
        print("SGD Recall Score:\n " , sgd_recall, "\n")
        sgd_f_score = f1_score(y_train_5, y_train_pred_90) 
        print("SGD F Score:\n", sgd_f_score, "\n")
        
        #ROC CURVE
        # The dotted line represents the ROC curve of a purely random clasifier: 
        # A good classsifier staays as far away from that line as possible(
        #the top left corner)
        # Compare classifiers by measuring the area under the curve
        # Perfect classifier will have a ROC AUC = 1, while a random 
        # classifier will equal 0.5
        fpr, tpr, thresholds = roc_curve(y_train_5, sgd_y_scores)
        
        def plot_roc_curve(fpr, tpr, label = None):
            plt.plot(fpr, tpr, linewidth=2, label=label)
            plt.plot([0,1], [0,1], 'k--') # dashboard diagonal
            plt.show()
            
        plot_roc_curve(fpr, tpr, "Stochastic Gradient Descent ROC Curve")
        sgd_roc_score = roc_auc_score(y_train_5, sgd_y_scores)
        print("SGD ROC curve score:\n",sgd_roc_score, "\n")
        return fpr, tpr, thresholds  
        
    #Random Forest Classifier
    def evaluateBinaryRFC(x_train, y_train_5, five):
        forest_clf = RandomForestClassifier(random_state = 42)
        forest_clf.fit(x_train, y_train_5)
        pred_5 = forest_clf.predict([five])
        print("RFC Model prediction for image:\n", pred_5, "\n")
        y_probas_forest =cross_val_predict(forest_clf, x_train, y_train_5,
                                           cv = 3, method = "predict_proba")
        rfc_accuracy = cross_val_score(forest_clf, x_train,
                                       y_train_5, cv=3, scoring = "accuracy")
        print("RFC Accuracy:\n", rfc_accuracy, "\n")
        y_scores_forest = y_probas_forest[:,1] # score = proba of positive class
        fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, 
                                                             y_scores_forest)
    
        # Error analysis
        y_train_pred = cross_val_predict(forest_clf, x_train, y_train_5, cv=3)
        conf_mx = confusion_matrix(y_train_5, y_train_pred)
        rfc_precision = precision_score(y_train_5, y_train_pred)
        print("Precision score for RFC:\n", rfc_precision, "\n" )
        rfc_recall = recall_score(y_train_5, y_train_pred)
        print("RFC Recall Score:\n " , rfc_recall, "\n")
        rfc_f_score = f1_score(y_train_5, y_train_pred) 
        print("RFC F Score:\n", rfc_f_score, "\n")
        print("Confusion Matrix for RFC: \n", conf_mx)
        
        def plot_roc_curve(fpr, tpr, label = None):
            plt.plot(fpr, tpr, linewidth=2, label=label)
            plt.plot([0,1], [0,1], 'k--') # dashboard diagonal
            plt.show()
        
        plot_roc_curve(fpr_forest, tpr_forest, "Random Forest ROC Curve")
    
        # ROC_AUC_SCORE 
        score = roc_auc_score(y_train_5, y_scores_forest)
        print("ROC score for random forest:\n", score, "\n")
        return fpr_forest, tpr_forest
        
    def evaluateBinarySVM(x_train, y_train_5, five):
        Gamma = 0.001
        C = 1
        svm_clf = svm.SVC(kernel = 'poly' , C = C, gamma = Gamma)
        svm_clf.fit(x_train, y_train_5)
        pred_5 = svm_clf.predict(([five]))
        print("SVM Model prediction for image:\n", pred_5, "\n")
        #Accuracy rating
        svm_accuracy = cross_val_score(svm_clf, x_train,
                                       y_train_5, cv=3, scoring = "accuracy")
        print("SVM Accuracy:\n", svm_accuracy, "\n")
        # Confusion Matrix
          # Error analysis
        y_train_pred = cross_val_predict(svm_clf, x_train, y_train_5, cv=3)
        conf_mx = confusion_matrix(y_train_5, y_train_pred)
        svm_precision = precision_score(y_train_5, y_train_pred)
        print("Precision score for SVM:\n", svm_precision, "\n" )
        svm_recall = recall_score(y_train_5, y_train_pred)
        print("SVM Recall Score:\n " , svm_recall, "\n")
        rfc_f_score = f1_score(y_train_5, y_train_pred) 
        print("SVM F Score:\n", rfc_f_score, "\n")
        print("Confusion Matrix for SVM: \n", conf_mx)
        svm_y_scores = cross_val_predict(svm_clf, x_train, y_train_5, cv=3, 
                             method="decision_function")
        
        #ROC CURVE
        # The dotted line represents the ROC curve of a purely random clasifier: 
        # A good classsifier staays as far away from that line as possible(
        #the top left corner)
        # Compare classifiers by measuring the area under the curve
        # Perfect classifier will have a ROC AUC = 1, while a random 
        # classifier will equal 0.5
        fpr, tpr, thresholds = roc_curve(y_train_5, svm_y_scores)
        def plot_roc_curve(fpr, tpr, label = None):
            plt.plot(fpr, tpr, linewidth=2, label=label)
            plt.plot([0,1], [0,1], 'k--') # dashboard diagonal
            plt.show()
        plot_roc_curve(fpr, tpr, "Support Vector Machine ROC Curve")
        svm_roc_score = roc_auc_score(y_train_5, svm_y_scores)
        print("SVM ROC curve score:\n",svm_roc_score, "\n")
        return fpr, tpr, thresholds
        
    def stochasticGradientDescentClassifier(x_train, y_train, x_test, y_test):
        clf = SGDClassifier(random_state = 42)
        clf.fit(x_train, y_train)
        
        # SGD Accuracy rating
        accuracy = cross_val_score(clf, x_train,
                                       y_train, cv=3, scoring = "accuracy")
        print("Stochastic Gradient Descent Accuracy:\n", accuracy, "\n")
        
        pred_y = clf.predict(x_test)
        report = classification_report(y_test, pred_y)
        print("Printing SGD classification report ...\n")
        print(report)
        
        # Confusion Matrix 
        conf_mx = confusion_matrix(y_test, pred_y)
        print("Confusion Matrix ...\n", conf_mx)
        print("Heated confusion matrix")
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.show()
        
        # Visualization of the errors
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        np.fill_diagonal(norm_conf_mx, 0)
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        plt.show()
        
        

    def randomForestClassifier(x_train, y_train, x_test, y_test):
        clf = RandomForestClassifier(random_state = 42)
        clf.fit(x_train, y_train)
        
        # RFC Accuracy rating 
        accuracy = cross_val_score(clf, x_train,
                                       y_train, cv=3, scoring = "accuracy")
        print("Random Forest Accuracy:\n", accuracy, "\n")
        
        pred_y = clf.predict(x_test)
        report = classification_report(y_test, pred_y)
        print("Printing RFC classification report ...\n")
        print(report)
        
        # Confusion Matrix 
        conf_mx = confusion_matrix(y_test, pred_y)
        print("Confusion Matrix ...\n", conf_mx)
        print("Heated confusion matrix")
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.show()
        
        # Visualization of the errors
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        np.fill_diagonal(norm_conf_mx, 0)
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        plt.show()
        
        
    def supportVectorMachineClassifier(x_train, y_train, x_test, y_test):
        Gamma = 0.001
        C = 1
        clf = svm.SVC(kernel = 'poly' , C = C, gamma = Gamma)
        clf.fit(x_train, y_train)
        
        #Accuracy rating
        accuracy = cross_val_score(clf, x_train,
                                       y_train, cv=3, scoring = "accuracy")
        print("SVM Accuracy:\n", accuracy, "\n")
        
        pred_y = clf.predict(x_test)
        report = classification_report(y_test, pred_y)
        print("Printing SVM classification report ...\n")
        print(report)
        
        # Confusion Matrix 
        conf_mx = confusion_matrix(y_test, pred_y)
        print("Confusion Matrix ...\n", conf_mx,"\n")
        print("Heated confusion matrix...")
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.show()
        
        # Visualization of the errors
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        np.fill_diagonal(norm_conf_mx, 0)
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        plt.show()
        
    def testBestModel():
        try:
            train_file = "train.csv"
            test_file = "test.csv"
            test_data = pd.read_csv(test_file)
            train_data = pd.read_csv(train_file)
            print(test_file, train_file,"loaded succesfully")
        except Exception as e:
            print("Error occured {}".format(e))
            
        # Get data ready 
        y_train = train_data.iloc[:,0]
        x_train = train_data.drop("label", axis=1)
        x_test = test_data
        
        # Train SVM Classifier
        Gamma = 0.001
        C = 1
        clf = svm.SVC(kernel = 'poly' , C = C, gamma = Gamma)
        clf.fit(x_train, y_train)
        
        #Accuracy rating
        accuracy = cross_val_score(clf, x_train,
                                       y_train, cv=3, scoring = "accuracy")
        print("SVM Accuracy:\n", accuracy, "\n")
        
        predictions = clf.predict(x_test)
        y_train_pred = cross_val_predict(clf, x_train, y_train, cv = 3)
        report = classification_report(y_train, y_train_pred)
        print("Printing SVM classification report ...\n")
        print(report)
        
         # Confusion Matrix 
        conf_mx = confusion_matrix(y_train, y_train_pred)
        print("Confusion Matrix ...\n", conf_mx,"\n")
        print("Heated confusion matrix...")
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.show()
        
        # Visualization of the errors
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        np.fill_diagonal(norm_conf_mx, 0)
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        plt.show()
        
        # Test file for Kaggle 
        pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),
              "Label":predictions}).to_csv("Submission.csv", index = False)

    # main
    if __name__ == "__main__":
     
        try:
            MNIST_FILE = "MNIST.csv"
            data = pd.read_csv(MNIST_FILE)
            print(MNIST_FILE, "loaded successfully\n")
        except Exception as e:
            print("Error occured {}".format(e))
        
        #to evaluate how each model compares with a binary classifier
        x_train_binary_classifier = data.drop(
                "label", axis = 1)
        print(x_train_binary_classifier.shape, "\n")
        
        #each image has 784 features. This is because each image is 28x28 pixals
        # I am testing a single digit 5 to evaluate each models performance 
        # before testing the whole set of numbers.
        x_train_binary_classifier = x_train_binary_classifier.to_numpy()
        five = x_train_binary_classifier[8]
        five_image = five.reshape(28,28)
        print("First I am going to test and train my models on digit = 5" )
        plt.title("Handwritten image = 5")
        plt.imshow(five_image, cmap = "binary")
        plt.axis("off")
        plt.show()
   
        # Split the Data into test and training data 
        x_train, y_train, x_test, y_test = splitData(data)
        y_train_5 = (y_train == 5)
        # Shape of data sets
        print("Shape of data sets")
        print("x train data shape = ", x_train.shape)
        print("y train data shape = ", y_train.shape)
        print("x test data shape = ", x_test.shape)
        print("y test data shape = ", y_test.shape, "\n")
        
        
        # Initial performance measures of Stochastic Gradient Descent model
        # when image = 5
        print("Evaluating performce of SGD Model for image = 5 ...")
        sgd_fpr, sgd_tpr, sgd_thresholds = evaluateBinarySGD(x_train, 
                                                            y_train_5, five)
        # Initial performance measure of Random forest model when image = 5
        print("Evaluating performce of RFC Model for image = 5 ...")
        fpr_forest, tpr_forest = evaluateBinaryRFC(x_train, y_train_5, five)
        plt.title("Randomo Forest Roc Curve")
        #Intital performance measure of SVM when image = 5
        print("Evaluating performance of SVM Model for image = 5 ...")
        svm_fpr, svm_tpr, svn_thresholds = evaluateBinarySVM(x_train, 
                                                             y_train_5, five)
        
        # ROC curve comparisons 
        print("ROC Curve Comparisons")
        plt.plot(fpr_forest, tpr_forest, linewidth = 2, 
                 label =  "Random Forest")
        plt.plot(svm_fpr, svm_tpr, label = "SVM")
        plt.plot(sgd_fpr, sgd_tpr, label = "SGD")
        plt.title("ROC Curve Comparison")
        plt.legend(loc = "lower right")
        plt.show
        
        #plot digits
        plt.figure(figsize=(9,9))
        example_images = x_train[:100]
        plot_digits(example_images, images_per_row=10)
        plt.show()
        
        print("Evaluating perfomance of SGD model for all images ...")
        stochasticGradientDescentClassifier(x_train, y_train, x_test, y_test)
        
        print("Evaluating perfomance of RFC model for all images ...")
        randomForestClassifier(x_train, y_train, x_test, y_test)
        
        print("Evaluating perfomance of SVM model for all images ...")
        supportVectorMachineClassifier(x_train, y_train, x_test, y_test)
        
        print("Further Testing using Kaggles train and test files ...\n")
        print("Evaluating performance of SVM model for all images ...")
        testBestModel()
       
        
        
        
        
        
        
        
        
        
        
        
        
        