#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from time import time
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random


def SetLogger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)


def extract_label(record):
    label = record[-1]
    return float(label)


def convert_float(x):
    return 0 if x == "NULL" else float(x)

#[3, 4, 5, 8, 9, 15, 16, 25]
def extract_features(record):
    #index = np.arange(51)
    features_index = [4, 8, 9, 10, 11, 12, 16, 26]
    #features_index = index[2:51]
    features = [convert_float(record[x]) for x in features_index]
    return np.asarray(features)



def prepare_data(sc):
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile("./data.csv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    lines = rawData.map(lambda x: x.split(","))
    print (lines.first())
    print("共計：" + str(lines.count()) + "筆")
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r)))

    (trainData, validationData,
     testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("將資料分trainData:" + str(trainData.count()) +
          "validationData:" + str(validationData.count()) +
          "testData:" + str(testData.count()))
    return (trainData, validationData, testData)


def CreateSparkContext():
    sparkConf = SparkConf() \
        .setAppName("RunRandomForestClassficaiton") \
        .set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=sparkConf)
    print ("master=" + sc.master)
    SetLogger(sc)
    return (sc)


def trainEvaluateModel(trainData):
    model = RandomForest.trainClassifier(trainData, 2, categoricalFeaturesInfo={},
                                     numTrees=15, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=12, maxBins=32)
    return model


def evaluateModel(model, valiadationData):
    score = model.predict(validationData.map(lambda p: p.features))
    labelsAndPredictions = validationData.map(lambda p: (p.label)).zip(score)
   
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(validationData.count())

    print('Test Error = ' + str(testErr))
    print('Learned classification tree model:')
    print(model.toDebugString())


def rocCurve(model, testData):

   # Compute raw scores on the test set
    predictions = model.predict(testData.map(lambda x: x.features))
    labels_and_predictions = testData.map(lambda x: x.label).zip(predictions)
    acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(testData.count())
    print("Model accuracy: %.3f%%" % (acc * 100))

    # Instantiate metrics object
    start_time = time()
    metrics = BinaryClassificationMetrics(labels_and_predictions)

    # Area under precision-recall curve
    print("Area under Precision/Recall (PR) curve: %.f %%" % (metrics.areaUnderPR * 100))

    # Area under ROC curve
    print("Area under Receiver Operating Characteristic (ROC) curve: %.3f %%" % (metrics.areaUnderROC * 100))

    end_time = time()
    elapsed_time = end_time - start_time
    print("Time to evaluate model: %.3f seconds" % elapsed_time)

def predictData(sc, model):
    #----------------------1.匯入並轉換資料-------------
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile("./data.csv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x:x !=header)
    lines = rawData.map(lambda x: x.split(","))
    print("共計：" + str(lines.count()) + "筆")
    #----------------------2.建立訓練評估所需資料 LabeledPoint RDD-------------
    #labelpointRDD = lines.map(lambda r: LabeledPoint("0.0", extract_features(r)))
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r)))

    #----------------------4.進行預測並顯示結果--------------

    # 把預測結果寫出來
    f = open('workfile', 'w')
    for lp in labelpointRDD.take(1000):
        predict = int(model.predict(lp.features))
        dataDesc = "  " + str(predict) + " "
        f.write(dataDesc)
    f.close()


if __name__ == "__main__":

    sc = CreateSparkContext()
    (trainData, validationData, testData) = prepare_data(sc)
    trainData.persist()
    validationData.persist()
    testData.persist()
    modelBest = trainEvaluateModel(trainData)
    errLowest = evaluateModel(modelBest, testData)

    for i in range(3):
        (trainData, validationData, testData) = prepare_data(sc)
        trainData.persist()
        validationData.persist()
        testData.persist()
        model = trainEvaluateModel(trainData)
        err = evaluateModel(model, testData)
        if err <= errLowest:
            errLowest = err
            modelBest = model
            rocCurve(modelBest, validationData)
    predictData(sc, modelBest)
