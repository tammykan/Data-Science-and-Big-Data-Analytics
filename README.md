# Data-Science-and-Big-Data-Analytics

For NCCU 1052 Data Science and Big Data Analytics Final Project <br>
Finish a machine learning challenged hosted by Kaggle in conjunction with Expedia : [Expedia-personalized-sort](https://www.kaggle.com/c/expedia-personalized-sort)


###  Member 
1. [Liu Z.Y.](https://github.com/yad50968)
2. [Tammykan](https://github.com/tammykan)
3. [Cwsu](https://github.com/cwsu)

### What features I choose
1. visitor_hist_starrating
2. prpo_starrating
3. prop_review_score
4. prop_brand_bool
5. prop_location_score1
6. prop_location_score2
7. promotion_flag
8. orig_destination_distance

### What data I predict
1. booking_bool


### Preprocessing
After download the data, I seperated the booking_true and booking_false data.
And get the same number of rows with awk
```sh
    awk -F "," '{if(substr($54,0,1)=="1") {print}} ' data.csv > book.csv
    awk -F "," '{if(substr($54,0,1)=="0") {print}} ' data.csv > no_book.csv
    tail -n ? no_book.csv > no_book_tmp.csv
    cat book.csv no_book_tmp > data.csv
```

### Run
Use spark to run the python script
```sh
    spark-submit --master local ./Main.py
```

### Algorithm
I use RandomForest.trainClassifier and the null model is 1/2(guess)

### Result
1. Accuracy: 77.89%
2. Area under Precision/Recall (PR) curve: 87%
3. Area under ROC curve: 78.251%


### Reference
1. [Spark doc](https://spark.apache.org/docs/latest/)
2. [Personalize Expedia Hotel Searches - ICDM 2013](https://www.kaggle.com/c/expedia-personalized-sort)
3. [benhamner](https://github.com/benhamner/ExpediaPersonalizedSortCompetition)
