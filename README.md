# Clustering_InterestsSurvey
Implemented dimensionality reduction for 217 features for community clustering analysis.

## Summary
- DataSet: 217 interests, 6340 persons.
- Data pre-processing：
  - Solve the problem of missing values ​​and outliers to avoid errors in cluster analysis.
  - For example, exclude people with too many or too few interests.
- Dimensionality reduction：
  1. Low variance filter, Subset selection
    - for clustering, more concentrated the distribution of the data corresponding to the feature, the better the contribution to the classifier.
    - Because the type of feature data set is 1 or 0, like Boolean data, and Boolean features are Bernoulli distribution (The variance is p*(1-p)).
    - Filtering on the variance will remove features that have a value of 0 or 1 in more than 80% of the samples.
  2. Principal Component Analysis
    - Because the dataset is not labeled data, PCA is more suitable than LDA as a dimensionality reduction method.
    - find a projection axis can be obtained after projection to obtain the maximum variation of this group of data.
- Clustering：
  - The K-means clustering algorithm (K-means) was selected as this method.
  - Reason：
    - Because the data is unlabeled, it is necessary to use unsupervised learning. 
    - The outliers have been excluded from data preprocessing, so using this algorithm will not cause noise data to be affected.
  - When K < 4, the curve drops sharply; when K > 4, the curve tends to be stable, so the inflection point 4 is K the best value.
- Result Interpretation：
  - There are four groups in total.
  - Perform an interest analysis for each group, according to their pattern on the principal components.
