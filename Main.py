#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    features_index = [3, 4, 5, 14, 15, 16, 25, -2, -3]
    features = [convert_float(record[x]) for x in features_index]
    return np.asarray(features)


def prepare_data(sc):
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile("./train_100mb.csv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    lines = rawData.map(lambda x: x.split(","))
    print (lines.first())
    print("共計：" + str(lines.count()) + "筆")
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r)))

    (trainData, validationData,
     testData) = labelpointRDD.randomSplit([3, 0, 7])
    print("將資料分trainData:" + str(trainData.count()) +
          "validationData:" + str(validationData.count()) +
          "testData:" + str(testData.count()))
    return (trainData, validationData, testData)


def CreateSparkContext():
    sparkConf = SparkConf()   \
        .setAppName("RunDecisionTreeRegression") \
        .set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=sparkConf)
    print ("master=" + sc.master)
    SetLogger(sc)
    return (sc)

def trainEvaluateModel(trainData,impurity,maxDepth, maxBins):

    model = DecisionTree.trainClassifier(trainData, numClasses=2, categoricalFeaturesInfo={}, impurity=impurity, maxDepth=maxDepth, maxBins=maxBins)
    return model
def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    labelsAndPredictions = validationData.map(lambda p: p.label).zip(score)
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(validationData.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification tree model:')
    print(model.toDebugString())
if __name__ == "__main__":
    sc = CreateSparkContext()
    (trainData, validationData, testData) = prepare_data(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    model = trainEvaluateModel(trainData, "gini", 10, 100)
    evaluateModel(model, testData)
