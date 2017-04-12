#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
from time import time
import numpy as np
import matplotlib.pyplot as plt
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
    features_index = [3, 4, 5, 14, 15, 16, 25]
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
    labelpointRDD = lines.map(lambda r: LabeledPoint(1.0, extract_features(r)))

    print labelpointRDD.first()
    (trainData, validationData,
     testData) = labelpointRDD.randomSplit([8, 1, 1])
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


if __name__ == "__main__":
    sc = CreateSparkContext()
    (trainData, validationData, testData) = prepare_data(sc)
