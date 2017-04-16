#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from time import time
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics


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


def extract_features(record):
    '''
        3 : visitor_location_country_id
        4 : visitor_hist_starrating
        5 : visitor_hist_adr_usd
        15 : price_use
        16 : promotion_flag
        25 : srch_query_affinity_score
    '''

    features_index = [3, 4, 5, 15, 16, 25]
    features = [convert_float(record[x]) for x in features_index]
    return np.asarray(features)



def prepare_data(sc):
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile("./train_100mb.csv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    lines = rawData.map(lambda x: x.split(","))
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r)))

    (trainData, validationData,
     testData) = labelpointRDD.randomSplit([3, 0, 7])
    print("將資料分trainData:" + str(trainData.count()) +
          " validationData:" + str(validationData.count()) +
          " testData:" + str(testData.count()))
    return (trainData, validationData, testData)


def CreateSparkContext():
    sparkConf = SparkConf()   \
        .setAppName("RunDecisionTreeClassficaiton") \
        .set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=sparkConf)
    SetLogger(sc)
    return (sc)


def trainEvaluateModel(trainData, impurity, maxDepth, maxBins):

    model = DecisionTree.trainClassifier(trainData, numClasses=2, categoricalFeaturesInfo={}, impurity=impurity, maxDepth=maxDepth, maxBins=maxBins)
    return model


def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    labelsAndPredictions = validationData.map(lambda p: p.label).zip(score)
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(validationData.count())
    print('Test Error = ' + str(testErr))




def predictData(sc, model):
    #----------------------1.匯入並轉換資料-------------
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile("./train_100mb.csv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x:x !=header)
    lines = rawData.map(lambda x: x.split(","))
    #----------------------2.建立訓練評估所需資料 LabeledPoint RDD-------------
    labelpointRDD = lines.map(lambda r: LabeledPoint("0.0", extract_features(r)))

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
    model = trainEvaluateModel(trainData, "gini", 10, 100)
    evaluateModel(model, testData)
    predictData(sc, model)
