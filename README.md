# Preprocessing Directory

- Datasets Directory contains all of the original .csv dataset files along with the python script used to merge them together
- Imputation Directory contains the mutlivariate imputation script and the corresponding .csv files we used to fill in data points with missing features
- Feature Summarization Directory contains the script for summarizing each feature into three constituent features: it's 5-year mean, standard deviation, and trend.
- Final Dataset Directory contains the .csv file we used for our model
- Within the preprocessing directory we also have the script we used to merge our already merged datasets from the World Bank with our financial crisis label dataset. We also have a count_data_points.py file that we used to keep track of how many data points we had at each step of preprocessing.

# Model Directory
- This directory contains all of the code/data relevant to our random forest gmm, kmeans, logistic regression, and gradient boosting models along with the visualizations of each model's performance.
