# SberbankSegmentation Part 1 (Clustering)
Analysis of bank transactions 
ReadMe for Part 1 
# 1.	Data Observing

Initially, we were working on loading the datasets and observing them. Research depicts the outliers in transactions dataset called ‘data’. However, to make more precise calculations and analysis we converted the ‘sum’ column to the ‘abs_sum’ column with absolute values for all transactions. As the result, we got a boxplot without outliers based on IQR calculations.
Clearing such many outliers will lead to the loss of the essence of the entire clustering, so the decision was made to keep everything.

# 2.	Data cleaning

## 2.1.	Missing Values

This part is considering missing values if there are any of them. Based on the plot we could conclude that there are no missing values. Hence, it means that data was correctly scrapped.

## 2.2.	Drop duplicates 

Here we identified how many of the same records are in the dataset. The answer is 19 duplicates.

## 2.3.	Data with invalid code/types values

### 2.3.1.	 Removing rows if their type/code is not found in codes/types tables

If the ‘data’ dataset’s records do not consist of any of type/code from code/type tables they would be deleted

### 2.3.2.	 Removing rows if the description of their codes/types values are invalid in codes/types tables

The same step with removing but in this case, we were deleting descriptions of codes/types from ‘data’ if they would not be found in the codes/types table

### 2.3.3.	Due to the types table containing duplicates in the type description we should remove these duplicates and replace invalid types in general dataframe 

Removing duplicates from the types table and replacing the duplicates with unique ones which are the same as duplicates

### 2.3.4.	Removing duplicates and invalid data from types table

Types table has invalid data and many duplicates. So, we have decided to drop such duplicates in the types table

## 2.4.	New dataframe with values from types and codes tables

As we mentioned before, we made some cleaning in each dataset. Hence, we created a new version of the ‘data’ dataset without invalid information and duplicates

# 3.	Exploratory Data Analysis (EDA)

This part is essential for research in terms of performing preliminary investigation on data and its features to reveal patterns, detect anomalies, test hypotheses, verify assumptions, etc. 
Abnormal outliers were detected around day 200, so we are building another lineplot to determine if this was due to a withdrawal of money from the account or its replenishment.
As you can see, for transaction code 6010 and transaction type 7070, there are abnormal values for the number of transactions.
Exploring 195th day, we can see an outlier with the transaction with the sum of about 70 million
It seems that we have reached out to Putin.
Diving into the history of a client that made a huge transaction, affecting the whole 195th day, we can get access to his data

This plot represents transactions distribution by time. The majority of transactions were completed at midnight. Hence, we can suggest that it was the subscription to services(e.g. Apple Music, Netflix, Spotify, YandexPlus)

Since we put '00' (midnight) at 'morning' in the list 'the time of the day', the graph represents column 'morning' as the highest one because of the subscriptions.

# 4.	Feature Engineering

Feature Engineering produces new features for both supervised and unsupervised learning, with the goal of simplifying and speeding up data transformations while also enhancing model accuracy.

## 4.1.	Absolute sum of transactions (abs_sum) and account balance (sum)

Working on a sum of client’s transactions, creating absolute value for transactions 

## 4.2.	Number of transactions (n_transactions)

Calculating the number of transactions for each client 

## 4.3.	Average transaction (average_transaction)

In this stage, we found out the average absolute sum of transactions per client

## 4.4.	Gender (gender)

Dropped the duplicates and merged them with ‘data’, renamed target to gender

## 4.5.	Percentage from the total sum of transactions (percentage)

Here we defined the percentage of each client’s transactions from the whole transactions in dataframe

## 4.6.	Number of transactions for each cluster of code and type

### 4.6.1.	 Installing required library and its modules

Installing nltk package and stopwords

### 4.6.2.	 Lists of code and type descriptions for clusterization

Converting to list type_description and code_description

### 4.6.3.	 KMeans clusterization functions

Creating functions to make K means clusterization. The first function called process_text works with text in type and code descriptions, tokenize text. The second function cluster_text transforms text into TF-IDF coordinates and then cluster texts using K-means build-in function.  

### 4.6.4.	 Adding new features which got from description clusterization (cluster <code/type> number)

The function counts the number of transactions in each related cluster per each client to identify the special clients in further research

## 4.7.	Time of registration in minutes (time_of_registration) and the difference between transactions in minutes (transaction_frequecy)

Creating dataframe that consists of datetime and client_id then split date from the time. The next step was creating a time of registration. Finding out transaction frequency of user by calculating time difference between first and last transactions of the client.

## 4.8.	RFM features - Recency, Frequency, Monetary values

RFM analysis is a technique used to quantitatively rank and group customers based on the recency, frequency, and monetary total of their recent transactions to identify the best customers and perform targeted marketing campaigns. By calculating RFM we found out recency, frequency, and monetary for each unique client in dataframe.

## 4.9.	Results of Feature Engineering
As the result in the data_to_clusterization dataset, we got the table with all features made in previous steps related to unique users. Also, it could help to make more accurate segmentation regarding to clients.

# 5.	Data Preprocessing

## 5.1.	Non-linear transformation. Mapping to a Uniform distribution (Quantile transformer)

Quantile transforms are a technique for transforming numerical input or output variables to have a Gaussian or uniform probability distribution.

## 5.2.	Normalization

Normalize samples individually to unit norm.

## 5.3.	Conclusion

Spending a huge amount of time on the selection and testing of possible combinations of standardization, non-linear transformation, normalization, and other preprocessing activities we have found through experience that the most appropriate course of action at this stage would be a non-linear transformation (Quantile transformer) and normalization (Normalizer). It is with this combination that machine learning models give the best result. This can be explained by the fact that both methods are robust to outliers.

# 6.	RFM Analysis

## 6.1.	Calculating feature scores.

Recency: the freshness of the customer activity (example: time since the last transaction has been made). Frequency: the frequency of the customer transactions (example: total time of transactions or average time between transactions). Monetary: the intention of customer to spend or purchasing power of the customer (example: total or average value of transactions)

## 6.2.	Segment customers to find lost and most active bank users

We can rank these customers by combining their individual R, F, and M rankings to arrive at an aggregated RFM score. This RFM score, is displayed in the table.

## 6.3.	Conclusion
We segmented customers according to their total RFM score. Each customer has their own calculated score which depends on his transaction activity and recency. A score can vary in the range of 3-12 and the larger score, the more valuable customer is. However, the total score itself cannot give descriptive information, that's why we decided to make an RFM score with respect to each feature and it ranges from '111' till '444', where each number accords to Recency, Frequency, Monetary features respectively. 

After calculating descriptive, string-like RFM scores, we can segment bank users by their activity, as we did a few cells ago. We got the most active and valuable, so-called 'golden' customers, who often use our bank and make transactions with big sums. These customers must be rewarded for their trust in a bank by giving them special offers or privileges.

We also found 'lost' customers that used to use our bank quite often and make large transactions, but for some reason, they stopped using our bank. Such customers must be involved in our bank again, by sending them to target messages with special offers or having phone calls.

# 7.	Model Selection

We are going to implement 2 of the most popular clustering algorithms: KMeans and Agglomerative Clustering. There will also be selected optimal hyperparameters, and the performance of the models will be evaluated by metrics. In the end, we will create a visualization of the results obtained and compare them

## 7.1.	KMeans clusterization

### 7.1.1.	 Choosing the optimal number of clusters
Sum Squared Error (SSE) is an accuracy measure where the errors are squared, then added. It is used to determine the accuracy of the forecasting model when the data points are similar in magnitude.

Cluster analysis is a statistical technique designed to find the “best fit” of consumers (or respondents) to a particular market segment (cluster). It does this by performing repeated calculations (iterations) designed to bring the groups (segments) in tighter/closer. If the consumers matched the segment scores exactly, the sum of squared error (SSE) 
would be zero = no error = a perfect match.

Important: The lower the SSE, then the more similar are the consumers in the segment.

In addition, we worked out on a bunch of different metrics such as: Silhouette Coefficient, Davies-Bouldin
The Calinski-Harabasz index - can be used to evaluate the model, where a higher Calinski-Harabasz score relates to a model with better defined clusters. The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
The Davies-Bouldin index can be used to evaluate the model, where a lower Davies-Bouldin index relates to a model with better separation between the clusters.

### 7.1.2.	 Conclusion

To conclude, based on the calculation, we found out that the most optimal number of clusters is 14

### 7.1.3.	 KMeans model

Here we are implementing our data from dataframe and assign number of clusters as we defined it is equal to 14

### 7.1.4.	 Cross-Validation Score

Cross-Validation is a very useful tool of a data scientist for assessing the effectiveness of the model, especially for tackling overfitting and underfitting. As you can see above, Cross-Validation Score is very small that says about well-chosen hyperparameters and good data preprocessing.

### 7.1.5.	 TSNE Visualization

Here you can observe how qualitatively the data is divided into clusters.

## 7.2.	Agglomerative Clustering

### 7.2.1.	 Agglomerative clustering model

Recursively merge pair of clusters of sample data; uses linkage distance. Based on experiments and relying on the same metrics as in KMeans clustering, it was determined that the best linkage for agglomerative clustering is "ward". 

### 7.2.2.	 Metrics

Exactly because of ward linkage, the model produces the greatest result in Silhouette, Calinski, and Davis Bouldin metrics.

### 7.2.3.	 Match with KMeans results

The main aim of making Agglomerative Clustering was to compare with KMeans clustering, as the result we got the results are 99% the same as those obtained with KMeans clustering.

### 7.2.4.	 Dendogram

### 7.2.5.	 TSNE Visualization

## 7.3.	Conclusion

In this section, we segmented customer data into 14 clusters. This number was chosen based on the conducted model selection, after which the models of KMeans and Agglomerative clustering algorithms were created. 

According to the assessment, it can be concluded that both algorithms give impressive results: there are no outliers and stratifications of clusters, their boundaries are very clearly defined, and they are able to characterize one or another feature of their group.

# 8.	Outcomes

Best customers: The 13s cluster represents the best customers according to their transactions. The thing we can notice here is that these customers have the biggest amount of money and their money flow is high as well. They have the same amount of transactions as the 9th cluster (61/61). The gender of these classes is males. 

The 9th cluster shows us information about customers who created their accounts earlier the best customers and use Sberbank for transactions more often than other customers. 

The 3rd cluster contains customers with a positive amount of money on their accounts, however, money on their accounts, the whole money flow, and amount of transactions are twice lower than the first two clusters. 

These 3 clusters are our best customers and use Sberbank functionality really often. We also can say they are the oldest customers of this bank. From code Clusters, it’s obvious golden customers to many transactions on code cluster 1 and code cluster 4 types of services or stuff and prefers to use type 4 clusters for transactions. 

Big spenders: The 7th cluster is people who have almost the lowest amount on their accounts but their total sum of transactions is the biggest. This cluster is really nearby to the best customers they are quite good, they do not do that many transactions (35 on average) but the amount of average transactions is bigger in comparison with other clusters. 

The 5th cluster has a lot in common with the 7th one. Spent lots of money, not that often, but by the big amount of money on average. 
These two clusters are good enough, might be for some reason they don’t do as many transactions as best customers, we can say at first they have registered later, at second during less time and less transaction number they spend more money in total. 

Average customers: The 1st cluster of customers are not that loyal; they do 20 transactions and are far away from our standards of the best customers. We can say that this cluster in total is in the middle. There is nothing to specify about it. But we still can take a bit more care of them. 

The 11th, 4th and 2nd clusters are also pretty average. We can’t say much about them, they have registered comparatively recently so we need to give them a bit more time.

Almost lost: In the 6th cluster, these customers do rare transactions and the average transactions of these customers are low as well. 

The 12th cluster with the smallest amount of transaction money. But at the same time, they do transactions more often than the other 3 clusters in our almost lost type. 

The 0th cluster is something in the middle between the 6th, 12th, and 8th, 10th clusters, we almost lost them so we need to pay more attention to them. 

The 8th cluster represents the customers we almost lost, interesting thing about it that this cluser majorly represented by males. Meanwhile all other male clusters are best customers according to their features. 

The 10th cluster customers do most rare transaction, and usage of time is the smallest as well.


# ReadMe Part 2 (Classification)

Analysis of bank transactions 


# 1.	Data Observing

Initially, we were working on loading the datasets and observing them. Research depicts the outliers in transactions dataset called ‘data’. However, to make more precise calculations and analysis we converted the ‘sum’ column to the ‘abs_sum’ column with absolute values for all transactions. As the result, we got a boxplot without outliers based on IQR calculations.
Clearing such many outliers will lead to the loss of the essence of the entire clustering, so the decision was made to keep everything.

# 2.	Data cleaning

2.1.	Missing Values

This part is considering missing values if there are any of them. Based on the plot we could conclude that there are no missing values. Hence, it means that data was correctly scrapped.

2.2.	Drop duplicates 
Here we identified how many of the same records are in the dataset. The answer is 19 duplicates.

2.3.	Data with invalid code/types values

2.3.1.	 Removing rows if their type/code is not found in codes/types tables

If the ‘data’ dataset’s records do not consist of any of type/code from code/type tables they would be deleted

2.3.2.	 Removing rows if the description of their codes/types values are invalid in codes/types tables

The same step with removing but in this case, we were deleting descriptions of codes/types from ‘data’ if they would not be found in the codes/types table

2.3.3.	Due to the types table containing duplicates in the type description we should remove these duplicates and replace invalid types in general dataframe 

Removing duplicates from the types table and replacing the duplicates with unique ones which are the same as duplicates

2.3.4.	Removing duplicates and invalid data from types table

Types table has invalid data and many duplicates. So, we have decided to drop such duplicates in the types table

2.4.	New dataframe with values from types and codes tables

As we mentioned before, we made some cleaning in each dataset. Hence, we created a new version of the ‘data’ dataset without invalid information and duplicates

# 3.	Exploratory Data Analysis (EDA)

This part is essential for research in terms of performing preliminary investigation on data and its features to reveal patterns, detect anomalies, test hypotheses, verify assumptions, etc. 
We are working on data paired with time
The first graph represents abs_sum based on time distribution between targets, as we can see there is almost no difference between targets, despite the outliers.
A further step is removing outliers and observing the previous graph without outliers
The graph below represents the abs_sum based on time distribution between targets without outliers, according to the graph target 1 is a bit higher among all 4 times of the day
Further steps allow removing outliers from average transactions and the number of transactions. So, these graphs give an opportunity to make a conclusion about transactions related to targets(genders)

# 4. Feature Engineering

Feature Engineering produces new features for both supervised and unsupervised learning, with the goal of simplifying and speeding up data transformations while also enhancing model accuracy.

4.1.	Absolute sum of transactions (abs_sum) and account balance (sum)

Working on a sum of client’s transactions, creating absolute value for transactions 

4.2.	Number of transactions (n_transactions)

Calculating the number of transactions for each client 

4.3.	Average transaction (average_transaction)

In this stage, we found out the average absolute sum of transactions per client

4.4.	Gender (gender)

Dropped the duplicates and merged them with ‘data’, renamed target to gender

4.5.	Percentage from the total sum of transactions (percentage)

Here we defined the percentage of each client’s transactions from the whole transactions in dataframe

4.6.	Number of transactions for each cluster of code and type

4.6.1.	 Installing required library and its modules

Installing nltk package and stopwords

4.6.2.	 Lists of code and type descriptions for clusterization
Converting to list type_description and code_description

4.6.3.	 Adding new features which got from description clusterization (cluster <code/type> number)
The function counts the number of transactions in each related cluster per each client to identify the special clients in further research

4.7.	Time of registration in minutes (time_of_registration) and the difference between transactions in minutes (transaction_frequecy)
Creating dataframe that consists of datetime and client_id then split date from the time. The next step was creating a time of registration. Finding out transaction frequency of user by calculating time difference between first and last transactions of the client.

4.8.	RFM features - Recency, Frequency, Monetary values
RFM analysis is a technique used to quantitatively rank and group customers based on the recency, frequency, and monetary total of their recent transactions to identify the best customers and perform targeted marketing campaigns. By calculating RFM we found out recency, frequency, and monetary for each unique client in dataframe.

4.9.	Results of Feature Engineering

4.10.	Type

4.10.1.	type - consists of data of counted types for 0 or for 1, then calculate the probability for both 1 and 0 and find the difference between them. If a value is positive then this type belongs to 1, if the value is negative the type belongs to 0.

4.11.	Codetype

4.11.1.	Multiplication of code and type because it was giving a high score of correlation.

4.12.	Code

4.12.1.	Consists data of counted codes for 0 or for 1, then calculate a probability for both 1 and 0 and find the difference between them. If the value is positive then this code belongs to 1, if the value is negative the code belongs to 0.
As the result in the data_to_classification dataset, we got the table with all features made in previous steps related to unique users. Also, it could help to make more accurate classification regarding to clients.

# 5.Model Selection

5.1.	KNN

Classifier implementing the k-nearest neighbor’s vote. Regarding the Nearest Neighbors algorithms, if it is found that two neighbors, neighbor k+1 and k, have identical distances but different labels, the results will depend on the ordering of the training data.

5.2.	Decision Tree

A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes).

5.3.	Random Forest

A random forest is a meta estimator that fits several decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

5.4.	GradientBoostingClassifier

GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. Binary classification is a special case where only a single regression tree is induced.

5.5.	AdaBoost

An AdaBoost [1] classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

5.6.	SVM

SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.

5.7.	MLPClassifier

This model optimizes the log-loss function using LBFGS or stochastic gradient descent.

# 6.Conclusion

The best and most precise results were obtained by data_c dataset, which is the merged from classifications and experiments. In addition, ensembles provided the most decent values. Precision, f1_score, recall also have been calculated to identify how does our model works.

