import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from time import time

import pyspark as ps
from pyspark.sql.types import *
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

class MovieRecommender(object):
    """Template class for a Movie Recommender system."""

    def __init__(self, local=False):
        """Constructs a MovieRecommender"""
        self.spark = ps.sql.SparkSession.builder \
              .master("local[4]") \
              .appName("Movie Reccomender") \
              .getOrCreate()
        self.sc = self.spark.sparkContext
        self.logger = logging.getLogger('reco-cs')
        self.users = self.sc.textFile('data/users.dat').map(lambda x: (int(x.split('::')[0]), x))
        self.movies = self.sc.textFile('data/movies.dat').map(lambda x: (int(x.split('::')[0]), x))
        self.local = local



    def fit(self, ratings):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'movie', 'rating', 'timestamp'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")

        #Save the training data for later use:
        self.training_data = ratings.copy()
        # self.training_data = ratings.toPandas()
        self.users_train_unique = self.training_data.user.unique()
        self.movies_train_unique = self.training_data.movie.unique()

        #Begin Transforming the data for fitting
        t0 = time()
        users = self.users
        movies = self.movies
        ratings = self.spark.createDataFrame(ratings.copy())

        # Maps the ratings df structure to that of the test data'a
        ratings = ratings.rdd.map(tuple)
        ratings = ratings.map(lambda x: '::'.join(x))
        ratings = ratings.map(lambda x: (int(x.split('::')[0]), x))
        self.ratings = ratings

        # Joins all the tables together for training
        joined = ratings.join(users)
        temp = joined.map(lambda x: '::'.join(x[1])).map(lambda x: (int(x.split('::')[1]), x))
        joined_full = temp.join(movies).map(lambda x: '::'.join(x[1]))

        # Removes the :: seperator from the RDD
        def split_to_cols(x):
            values = x.split('::')
            return (int(values[0]), int(values[1]), int(values[2]))

        # Not used but kept around because it could be
        def get_ratings(x):
            values = x.split('::')
            return (int(values[2]))

        # Turns the RDD into a DataFrame
        spark_df = joined_full.map(split_to_cols)

        schema = StructType([
            StructField("userID", IntegerType(), True),
            StructField("movieID", IntegerType(), True),
            StructField("rating", IntegerType(), True)])

        # Creates the proper train DataFrame for fitting
        train = self.spark.createDataFrame(spark_df, schema)

        # Instantiate the model (Alternating Least Squares)
        als = ALS(
                itemCol='movieID',
                userCol='userID',
                ratingCol='rating',
                nonnegative=True,
                regParam=0.4,
                maxIter=10,
                rank=14)

        # Creates the reccomender by fitting the training data
        self.recommender = als.fit(train)

        # Fit the model
        print('Model Created. Trainging....')
        self.recommender = als.fit(train)
        self.fitted = True

        self.logger.debug("finishing fit")
        print('DONE! ', time()-t0, ' seconds.')
        return(self)


    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """

        # test_df = requests.toPandas()
        self.test_df = requests.copy()

        #Filter down the request data
        self.old_old = test_df[(test_df.user.isin(self.users_train_unique))
                          & (test_df.movie.isin(self.movies_train_unique))]
        newish = test_df[~((test_df.user.isin(self.users_train_unique))
                         & (test_df.movie.isin(self.movies_train_unique)))]
        self.newish = newish

        #Split off the new users/movies:
        self.requests_new_movies = newish[(newish.user.isin(self.users_train_unique))
                                    & ~(newish.movie.isin(self.movies_train_unique))]

        self.requests_new_users = newish[~((newish.user.isin(self.users_train_unique))
                                    & ~(newish.movie.isin(self.movies_train_unique)))]

        requests = self.spark.createDataFrame(self.old_old)

        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.count()))
        t0 = time()
        users = self.users
        movies = self.movies

        # Gets the requests in the right shape
        requests = requests.rdd.map(tuple)
        requests = requests.map(lambda x: '::'.join(x))
        requests = requests.map(lambda x: (int(x.split('::')[0]), x))

        joined = requests.join(users)
        temp = joined.map(lambda x: '::'.join(x[1])).map(lambda x: (int(x.split('::')[1]), x))

        joined_full = temp.join(movies).map(lambda x: '::'.join(x[1]))

        def split_to_cols(x):
            values = x.split('::')
            return (int(values[0]), int(values[1]), int(values[2]))

        def get_ratings(x):
            values = x.split('::')
            return (int(values[2]))

        data_rdd = joined_full.map(split_to_cols)
        j_ratings = joined_full.map(get_ratings)

        schema = StructType([
            StructField("userID", IntegerType(), True),
            StructField("movieID", IntegerType(), True),
            StructField("rating", IntegerType(), True)])

        test = self.spark.createDataFrame(data_rdd, schema)

        self.logger.debug("finishing predict for recognized users and movies")

        print('Transforming...')
        output = self.recommender.transform(test)

        output = output.toPandas()
        output.drop('rating',axis=1,inplace=True)
        output.rename(columns={'userID':'user', 'movieID':'movie'}, inplace = True)
        print('DONE! ', time()-t0, ' seconds.')

        print("Sending the new users to different model..")
        t0 = time()
        self.new_user_pred = self.weighted_Recommendation()
        output = pd.concat([output,self.new_user_pred],axis=0)
        print('DONE! ', time()-t0, ' seconds.')


        print("Sending the new movies to different model..")
        t0 = time()
        if self.local == False:
            self.new_movie_pred = self.requests_new_movies.copy()
            self.new_movie_pred['prediction'] = 2.5
            output = pd.concat([output,self.new_movie_pred],axis=0)
#         else:
#             for


        print('DONE! ', time()-t0, ' seconds.')
        return(output)


    def weighted_Recommendation(self, is_sparse=False):
        pd.options.display.float_format = '{:,.2f}'.format
        training = self.training_data.copy()
        users_movies = self.requests_new_users

        if is_sparse:

            grouped_training = pd.DataFrame(np.full(len(training.columns),2.5))
            grouped_training['movie'] = np.array(training.columns)
            grouped_training['rating']= np.array(training.mean(axis = 0))
            grouped_training['vote']= np.array(training.count(axis = 0))
            grouped_training = grouped_training[['movie','rating','vote']]

        else:
            training['rating'] = training['rating'].astype(int)
            grouped_training = training.groupby('movie') \
                        .agg({'user':'size', 'rating':'mean'}) \
                        .rename(columns={'user':'vote','rating':'rating'}) \
                        .reset_index()


        # Calculate the minimum number of voters required to be in the chart
        m = grouped_training['vote'].quantile(0.5)

        # Filter out all qualified movies into a new DataFrame
        scorings = grouped_training.copy().loc[grouped_training['vote'] >= m]

        F = pd.merge(users_movies, scorings, on='movie', how='left')
        F['rating'].fillna(2.5, inplace=True)
        final = F[['user','movie','rating']]
        final.rename(columns={'rating':'prediction'},inplace=True,copy=False)

        return(final)

    def pred_on_similarity(df, similarity_matrix, userID, movieID, num_similar=10):
        '''
        GENERATE 1 PREDICTED VALUE OF AN UNSEEN MOVIE FOR AN EXISTING USER BASED ON THAT USER'S RATINGS OF THE MOST
        SIMILAR MOVIES TO THE MOVIE IN QUESTION.


        df : 'pandas dataframe with columns user(int), movie(int)
        similarity_matrix : square matrix pd.DataFrame of similarities
        userID : int : id of user in df
        movieID : int/str : id of movie in df
        num_similary : int : compare movie in question to *num_similar* number of other movies the user has rated.
        '''
        n = num_similar
        movieID = str(movieID)
        user = df[df.user == userID][['movie','rating']] #get user movies and ratings by the user in question
        m = similarity_matrix[movieID].reset_index() #get similarities for the movie in question
        m.columns = ['movie','similarity'] #rename columns for merge
        merged = m.merge(user, on='movie',how='inner') #merge movie similarities with ratings
        merged['product'] = merged.rating*merged.similarity #calculate rating*similarity

        #get top similarity value for normalizing
        sorted_sims = merged.similarity.sort_values(ascending=False)
        norm = sorted_sims[sorted_sims < 1].iloc[0]

        #sort by top similarities, take first n ratings*similarities, take average, normalize
        p = np.mean(merged.sort_values(by='similarity', ascending=False)['product'][:n])/norm
        return p

if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
