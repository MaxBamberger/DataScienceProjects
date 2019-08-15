# Movie Recommender System

I built a recommendation system based off data from the
[MovieLens dataset](http://grouplens.org/datasets/movielens/). It includes movie
information, user information, and the users' ratings. The algorithm's goal is to suggest movies to users!

The **movies data** and **user data** are in `data/movies.dat` and `data/users.dat`.

The **ratings data** can be found in `data/training.csv`. The users' ratings have been broken into a training and test set for you (to obtain the testing set, we have split the 20% of **the most recent** ratings).

**additional metadata** can be found by downloading [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7).

## The Solution: 

Using a [Non-Negative Matrx Factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) technique (run via pyspark) user-movie ratings are inferred based user's similarity to other users and movies's similarity to other movies. 

This works decently well to fill in recommendations, however it cannot be applied for new-movies and new-users. This is known as a cold-start issue. To remedy the cold-start issue, I use a popularity algorithm for new users and an NLP-based content similarity algorithm for new-movies. 

## Note on running the solution
`recommender.py` script relies on spark, you may want to use the script `run_on_spark.sh` to execute the code or 

In a terminal, use: run_on_spark.sh src/run.py with arguments to run the recommender.
To run in a notebook, use: `jupyspark.sh` to open a jupyter notebook
