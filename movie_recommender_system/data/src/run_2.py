
from recommender import MovieRecommender     # the class you have to develop
import pandas as pd

# Reading REQUEST SET from input file into pandas
request_data = pd.read_csv('data/requests.csv')
# request_data = spark.read.csv(path_requests_, header=True)

# Reading TRAIN SET from input file into pandas
train_data = pd.read_csv('data/training.csv')
# train_data = spark.read.csv(path_train_, header=True)

# Creating an instance of your recommender with the right parameters
reco_instance = MovieRecommender()

# fits on training data, returns a MovieRecommender object
model = reco_instance.fit(train_data)

# apply predict on request_data, returns a dataframe
result_data = model.transform(request_data)

result_data.to_csv('output_1', index=False)
