import pandas as pd
import numpy as np

from sklearn.preprocessing import (MultiLabelBinarizer,
                                   StandardScaler)
from sklearn.metrics import (precision_score, recall_score, roc_curve,
                             confusion_matrix, accuracy_score, roc_auc_score)
from sklearn.model_selection import (train_test_split,
                                    KFold, cross_val_score)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.class_weight import compute_sample_weight

from topic_model import TopicModel

from copy import copy, deepcopy
import datetime
import ast
import re
import pickle
import matplotlib.pyplot as plt
import sys
plt.style.use('ggplot')

#if making API requests:
# import time
# import timeit
# import requests
# import json


class BuildModel:
    '''
    BuildModel Class.
    Object with methods for transforming movie data into a
    form for training a machine learning classifier

    Type:  object

    inputs
    ------
    df_tmdb: pandas dataframe. TMDB raw data
    df_omdb: pandas dataframe. OMDB raw data

    methods
    -------
    transform_data()
    add_star_power()
    add_writer_power()
    add_director_power()
    last_movie_award()
    top_production_score()
    add_chem_factor()
    add_macro_trend()
    add_release_proximity()
    final_transform()
    fit(**kwargs)
    model.plot_performance()

    '''

    def __init__(self, df_tmdb, df_omdb):
        '''
        initialize raw dataset by merging the TMBD and OMDB dataset into one master.

        inputs
        ------
        df_tmdb: pandas dataframe. TMDB raw data
        df_omdb: pandas dataframe. OMDB raw data
        '''
        #Split out the revenue data for later use
        self.df_revenue_data = df_tmdb[df_tmdb['revenue'] > 0]

        #Merge and set the master raw dataframe
        df_omdb.rename(columns={'imdbID':'imdb_id'},inplace=True)
        df_master = pd.merge(df_tmdb,df_omdb,how='inner',on='imdb_id')
        self.df_master = df_master
        self.evaluate = None
        self.trained_model = None

    def transform_data(self):
        '''
        Main function for clearning the raw dataset.
        Trims the master data down into just the usable features of the model
        Creates a few new features.
        Changes to the right data types
        '''
        df1 = self.df_master.copy()

        #filter down the master dataframe to just the usable data (valid revenue and budget data)
        df1['budget'] = df1['budget'].astype(int)
        df1 = df1[(df1['budget'] > 0) & (df1['revenue'] > 0)]
        print('---filtered down the master dataframe into valid financial data: {} rows'.format(df1.shape[0]))

        #Create profit columnn and classifcations based on half revenue - budget
        df1['profit'] = (df1['revenue'])-(df1['budget']*3)
        n_successes = df1[df1['profit']>0].shape[0]
        n_failures = df1[df1['profit']<0].shape[0]
        n_hmmm = df1[df1['profit']==0].shape[0]

        df1["success?"] = df1['profit'].apply(lambda x : 1 if x > 0 else 0)

        print('---number of successes v. failures: {} v. {}'.format(
                    df1[df1['success?']==1].shape[0],
                    df1[df1['success?']==0].shape[0]))

        print('  --films with of positive profit: {}'.format(n_successes))
        print('  --films with a negative profit: {}'.format(n_failures))
        print('  --films that broke even: {}'.format(n_hmmm))

            #Make 'belongs_to_collection a true/false
        df1['belongs_to_collection'] =  df1['belongs_to_collection'] \
                                            .apply(lambda x : 1 if type(x) == str else 0)

        #Isolate the genre data into lists
        df1['genres'] =  df1['genres']\
                                            .apply(lambda x : ast.literal_eval(x))\
                                            .apply(lambda x : [genre['name'].lower().strip() for genre in x])


        #Isolate the production_companies data into lists and note the # of production companies
        df1['production_companies'] =  df1['production_companies']\
                                            .apply(lambda x : ast.literal_eval(x))\
                                            .apply(lambda x : [d['name'].lower().strip() for d in x])
        df1['n_prod_companies'] = df1['production_companies']\
                                            .apply(lambda x : len(x))

        #Isolate the production_countries data into lists and note the # of production countries
        df1['production_countries'] =  df1['production_countries']\
                                            .apply(lambda x : ast.literal_eval(x))\
                                            .apply(lambda x : [d['iso_3166_1'].lower().strip() for d in x])
        df1['n_prod_countries'] = df1['production_countries']\
                                            .apply(lambda x : len(list(x)))

        #Isolate the spoken_languages data into lists and note the # of spoken_languages
        df1['spoken_languages'] = df1['spoken_languages']\
                                        .apply(lambda x : ast.literal_eval(x))\
                                        .apply(lambda x : [d['iso_639_1'].lower().strip() for d in x])

        df1['n_spoken_languages'] = df1['spoken_languages']\
                                        .apply(lambda x : len(list(x)))
        print('---new columns created')

        #Transform to the right dtypes
        df1['imdb_id'] = df1['imdb_id'].astype(str)
        df1['original_language'] = df1['original_language'].astype(str)
        df1['overview'] = df1['overview'].astype(str)
        df1['release_date'] = pd.to_datetime(df1['release_date'])
        df1.dropna(subset=['runtime'],inplace=True,axis=0)
        df1['runtime'] = df1['runtime'].astype(int)
        df1['tagline'] = df1['tagline'].astype(str)
        df1['Actors'] = df1['Actors'].astype(str)
        df1['Actors'] = df1['Actors'].apply(lambda x : [a.lower().strip() for a in x.split(',')])
        df1.dropna(subset=['Director'],inplace=True,axis=0)
        df1['Director'] = df1['Director'].astype(str)
        df1['Director'] = df1['Director'].apply(lambda x: [a.lower().strip() for a in x.split(',')])
        df1['Genre'] = df1['Genre'].astype(str)
        df1['Genre'] = df1['Genre'].fillna('n/a')
        df1['Language'] = df1['Language'].astype(str)
        df1['Language'] = df1['Language'].apply(lambda x: [a.lower().strip() for a in x.split(',')])
        df1['Plot'] = df1['Plot'].astype(str)
        df1['Production'] = df1['Production'].astype(str)
        df1['Production'] = df1['Production'].apply(lambda x: [a.lower().strip() for a in x.split(',')])
        df1['Rated'] = df1['Rated'].astype(str)
        df1['Released'] = df1['Released'].astype(str)
        df1['Title'] = df1['Title'].astype(str)
        df1['Writer'] = df1['Writer'].astype(str)

        regex = re.compile(".*?\((.*?)\)")
        df1['Writer'] = df1['Writer'].astype(str)
        df1['Writer'] = df1['Writer'].apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x).split(','))\
                                     .apply(lambda x: [w.lower().strip(' ') for w in x])\
                                     .apply(lambda x: list(dict.fromkeys(x)))

        df1.dropna(subset=['Year'],inplace=True,axis=0)
        df1['Year'] = df1['Year'].apply(lambda x: str(x)[:4] )
        df1['Year'] = df1['Year'].astype(int)

        #drop the rows that have N/A in both columns
        df1.drop(index=df1[((df1['Released'].isna()) | (df1['Released'] == 'N/A')) &
                           (df1['release_date'].isna())].index,
                 inplace=True)

        #convert released to datetime for the instances where release_date is NA
        df1['Released'] = pd.to_datetime(df1[df1['release_date'].isna()]['Released'])

        #replace release_date with that date if the release_date is NA
        df1.loc[df1['release_date'].isna(),'release_date'] = df1.loc[df1['release_date'].isna(),'Released']

        # print(df1[df1['release_date'].isna()][['release_date','Released']])
        df1.drop(index=df1[(df1['Released'].isna()) & (df1['release_date'].isna())].index,inplace=True)
        # print(df1[df1['release_date'].isna()][['release_date','Released']])

        #Get rid of garbage:
        df1['Genre'].fillna('N/A',inplace=True)
        df1[~(df1['Genre'].str.contains('Game-Show')) &
            ~(df1['Genre'].str.contains('Talk-Show'))]
        df1['Genre'] = df1['Genre'].apply(lambda x: x.split(','))

        #FillNA imdbRatings and Votes
        df1['imdbRating'] = df1['imdbRating'].fillna(df1['popularity'])
        df1['imdbVotes'] = df1['imdbVotes'].fillna(1)

        #keep just these columns:
        df1 = df1[['id','belongs_to_collection','genres','imdb_id','original_language','overview',
                   'production_companies','production_countries','release_date','revenue','runtime','spoken_languages',
                   'tagline','title','profit','success?','Actors','Country','Director','Genre',
                   'Language','Plot','Production','Rated','Released','Title','Writer','Year','n_prod_companies',
                   'n_prod_countries','n_spoken_languages','imdbRating','imdbVotes','popularity','budget','Awards']]

        df1.reset_index(inplace=True,drop=True)
        print('---Complete!')
        self.df_master_transform = df1

    def add_star_power(self, inplace=True):
        '''
        Add a metric to the transformed data that measures a movies' 'star-power' score
        based on previous results
        '''
        if inplace == False:
            orig_self = deepcopy(self)

        df1 = self.df_master_transform.copy()
            #Create an 'actors population' matrix:
        df_pop_matrix = df1[['title','Actors','imdbRating','imdbVotes','popularity','success?','release_date']].copy()

        #Scale the voting population and create a 'score' variable:
        df_pop_matrix['imdbVotes'].fillna(0,inplace=True)
        df_pop_matrix['imdbVotes'] = df_pop_matrix['imdbVotes'].astype(str)
        df_pop_matrix['imdbVotes'] = df_pop_matrix['imdbVotes'].apply(lambda x: x.replace(',','') if x != 'N/A' else 0)
        df_pop_matrix['imdbVotes'] = df_pop_matrix['imdbVotes'].astype(float)
        s_ = np.std(df_pop_matrix['imdbVotes'])
        u_ = np.mean(df_pop_matrix['imdbVotes'])
        df_pop_matrix['votes_scaled'] = df_pop_matrix['imdbVotes'].apply(lambda x: (x - u_)/s_ + 1)
        df_pop_matrix['imdbRating'] = df_pop_matrix['imdbRating'].apply(lambda x: x if x != 'N/A' else 0)
        df_pop_matrix['imdbRating'] = df_pop_matrix['imdbRating'].astype(float)
        df_pop_matrix['score'] = df_pop_matrix['votes_scaled'] * df_pop_matrix['imdbRating'] * df_pop_matrix['success?']

    #Try --- using popularity score:
    #     df_pop_matrix['score'] = df_pop_matrix['popularity']
    #     df_pop_matrix['score'] = df_pop_matrix['score'].astype(float)
    #     df_pop_matrix = df_pop_matrix.drop(['popularity'],axis=1)

        #blow out the actors into a sparse one-hot matrix:
        mlb = MultiLabelBinarizer()
        ohe = pd.DataFrame(mlb.fit_transform(df_pop_matrix['Actors']),columns=mlb.classes_, index=df_pop_matrix.index)
        df_pop_matrix = pd.concat([df_pop_matrix,ohe],axis=1)
        df_pop_matrix.drop('Actors',axis=1,inplace=True)
        print('---actor-movie matrix initialized! Applying a score to each actor...')

    #     #if you want to filter down the matrix to just successes:
    #     df_pop_matrix = df_pop_matrix[df_pop_matrix['success?'] == 1]
    #     df_pop_matrix.drop('success?',axis=1,inplace=True)

        #Apply the score to each actor
        matrix = pd.DataFrame((df_pop_matrix['score'].values * df_pop_matrix.iloc[:,8:].values.T).T,
                                  columns=df_pop_matrix.columns[8:], index=df_pop_matrix.index)
        df_pop_matrix = pd.concat([df_pop_matrix.iloc[:,:8],matrix],axis=1)

        #Sort by release_date and take cumsum so the score is relative to the time..
        df_pop_matrix = df_pop_matrix.sort_values('release_date',ascending=True)
        df_pop_matrix.iloc[:,8:] = df_pop_matrix.iloc[:,8:].cumsum(axis=0)
        df_pop_matrix.iloc[:,8:] = df_pop_matrix.iloc[:,8:].shift(1)
    #     df_pop_matrix.set_index('title',inplace=True)
        print('---matrix shape: ',df_pop_matrix.shape)

        print('---Actor-Movie matrix done! Calculating a population score for each actor...')

        #Create a popularity score for each actor based on the column sums of the sparse matrix (df_pop)
        #
        df_actor_pop_score = pd.DataFrame(columns=['actor','popularity_score'])
        i = 0
        for col in df_pop_matrix.columns[8:]:
            df_actor_pop_score.loc[i,'actor'] = col
            df_actor_pop_score.loc[i,'popularity_score'] = df_pop_matrix[col].max()
            i+=1

        df_actor_pop_score.set_index('actor',inplace=True)
        print('---Actor population dataframe done! Adding metrics to main dataframe..')
        df1 = df1.sort_values('release_date',ascending=True)
        print('---main dataframe shape: ',df1.shape)
        for i, row in df1.iterrows():
            actors = row['Actors']
            star_power = 0

    #             star_power_actor = df_pop_matrix[df_pop_matrix['release_date'] < row['release_date']][actor].sum()
            star_power_actor = [df_pop_matrix.loc[i,actor] for actor in actors]
    #           star_power_actor = df_actor_pop_score.loc[actor,'popularity_score']
            star_power = np.median(star_power_actor)
        #     print('total star power of {}: {}'.format(row['title'],star_power))
            df1.loc[i,'star_power'] = star_power
        print('---Complete')

        self.df_master_transform = df1
        self.df_pop_matrix_actors = df_pop_matrix
        self.actor_pop_scores = df_actor_pop_score

        if inplace == False:
            new_self = deepcopy(self)
            self = orig_self
            return new_self

    def add_writer_power(self, inplace=True):
        '''
        Add a metric to the transformed data that measures the success of the movie's writer / staff
        based on previous results
        '''
        if inplace == False:
            orig_self = deepcopy(self)

        df1 = self.df_master_transform.copy()

        #Create an 'actors population' matrix:
        df_pop_matrix = df1[['title','Writer','imdbRating','imdbVotes','popularity','success?','release_date']].copy()


        #Scale the voting population and create a 'score' variable:
        df_pop_matrix['imdbVotes'].fillna(0,inplace=True)
        df_pop_matrix['imdbVotes'] = df_pop_matrix['imdbVotes'].astype(str)
        df_pop_matrix['imdbVotes'] = df_pop_matrix['imdbVotes'].apply(lambda x: x.replace(',','') if x != 'N/A' else 0)
        df_pop_matrix['imdbVotes'] = df_pop_matrix['imdbVotes'].astype(float)
        s_ = np.std(df_pop_matrix['imdbVotes'])
        u_ = np.mean(df_pop_matrix['imdbVotes'])
        df_pop_matrix['votes_scaled'] = df_pop_matrix['imdbVotes'].apply(lambda x: (x - u_)/s_ + 1)
        df_pop_matrix['imdbRating'] = df_pop_matrix['imdbRating'].apply(lambda x: x if x != 'N/A' else 0)
        df_pop_matrix['imdbRating'] = df_pop_matrix['imdbRating'].astype(float)
        df_pop_matrix['score'] = df_pop_matrix['votes_scaled'] * df_pop_matrix['imdbRating'] * df_pop_matrix['success?']

    #Try --- using popularity score:
    #     df_pop_matrix['score'] = df_pop_matrix['popularity']
    #     df_pop_matrix['score'] = df_pop_matrix['score'].astype(float)
    #     df_pop_matrix = df_pop_matrix.drop(['popularity'],axis=1)

        #blow out the actors into a sparse one-hot matrix:
        mlb = MultiLabelBinarizer()
        ohe = pd.DataFrame(mlb.fit_transform(df_pop_matrix['Writer']),columns=mlb.classes_, index=df_pop_matrix.index)
        df_pop_matrix = pd.concat([df_pop_matrix,ohe],axis=1)
        df_pop_matrix.drop('Writer',axis=1,inplace=True)
        print('---Writer-Movie matrix initialized! Applying a score to each writer...')


        #if you want to filter down the matrix to just successes:
    #     df_pop_matrix = df_pop_matrix[df_pop_matrix['success?'] == 1]
    #     df_pop_matrix.drop('success?',axis=1,inplace=True)


        #Apply the score to each writer
        matrix = pd.DataFrame((df_pop_matrix['score'].values * df_pop_matrix.iloc[:,7:].values.T).T,
                                  columns=df_pop_matrix.columns[7:], index=df_pop_matrix.index)
        df_pop_matrix = pd.concat([df_pop_matrix.iloc[:,:7],matrix],axis=1)


        #Sort by release_date and take cumsum so the score is relative to the time..
        df_pop_matrix = df_pop_matrix.sort_values('release_date',ascending=True)
        df_pop_matrix.iloc[:,7:] = df_pop_matrix.iloc[:,7:].cumsum(axis=0)
        df_pop_matrix.iloc[:,7:] = df_pop_matrix.iloc[:,7:].shift(1)
    #     df_pop_matrix.set_index('title',inplace=True)
        print('---matrix shape: ',df_pop_matrix.shape)
        print('---Writer-Movie matrix done! Calculating a population score for each writer...')


        #Create a popularity score for each actor based on the column sums of the sparse matrix (df_pop)
        df_writer_pop_score = pd.DataFrame(columns=['Writer','popularity_score'])
        i = 0
        for col in df_pop_matrix.columns[7:]:
            df_writer_pop_score.loc[i,'writer'] = col
            df_writer_pop_score.loc[i,'popularity_score'] = df_pop_matrix[col].max()
            i+=1

        df_writer_pop_score.set_index('writer',inplace=True)
        print('---writer population dataframe done! Adding metrics to main dataframe..')
        df1 = df1.sort_values('release_date',ascending=True)
        print('---main dataframe shape: ',df1.shape)
        for i, row in df1.iterrows():
            writers = row['Writer']
            star_power = 0
            star_power_writer = [df_pop_matrix.loc[i,writer] for writer in writers]
            star_power = np.max(star_power_writer)
            df1.loc[i,'writer_power'] = star_power

        print('---Complete')
        self.df_master_transform = df1
        self.df_pop_matrix_writers = df_pop_matrix
        self.writer_pop_scores = df_writer_pop_score

        if inplace == False:
            new_self = deepcopy(self)
            self = orig_self
            return new_self

    def add_director_power(self, inplace=True):
        '''
        Add a metric to the transformed data that measures the success of the director / directors
        based on previous results
        '''
        if inplace == False:
            orig_self = deepcopy(self)

        df1 = self.df_master_transform.copy()

        #Create an 'actors population' matrix:
        df_pop_matrix = df1[['title','Director','imdbRating','imdbVotes','popularity','success?','release_date']].copy()

        #Scale the voting population and create a 'score' variable:
        df_pop_matrix['imdbVotes'].fillna(0,inplace=True)
        df_pop_matrix['imdbVotes'] = df_pop_matrix['imdbVotes'].astype(str)
        df_pop_matrix['imdbVotes'] = df_pop_matrix['imdbVotes'].apply(lambda x: x.replace(',','') if x != 'N/A' else 0)
        df_pop_matrix['imdbVotes'] = df_pop_matrix['imdbVotes'].astype(float)
        s_ = np.std(df_pop_matrix['imdbVotes'])
        u_ = np.mean(df_pop_matrix['imdbVotes'])
        df_pop_matrix['votes_scaled'] = df_pop_matrix['imdbVotes'].apply(lambda x: (x - u_)/s_ + 1)
        df_pop_matrix['imdbRating'] = df_pop_matrix['imdbRating'].apply(lambda x: x if x != 'N/A' else 0)
        df_pop_matrix['imdbRating'] = df_pop_matrix['imdbRating'].astype(float)
        df_pop_matrix['score'] = df_pop_matrix['votes_scaled'] * df_pop_matrix['imdbRating'] * df_pop_matrix['success?']

    #Try --- using popularity score:
    #     df_pop_matrix['score'] = df_pop_matrix['popularity']
    #     df_pop_matrix['score'] = df_pop_matrix['score'].astype(float)
    #     df_pop_matrix = df_pop_matrix.drop(['popularity'],axis=1)

        #blow out the actors into a sparse one-hot matrix:
        mlb = MultiLabelBinarizer()
        ohe = pd.DataFrame(mlb.fit_transform(df_pop_matrix['Director']),columns=mlb.classes_, index=df_pop_matrix.index)
        df_pop_matrix = pd.concat([df_pop_matrix,ohe],axis=1)
        df_pop_matrix.drop('Director',axis=1,inplace=True)
        print('---Director-Movie matrix initialized! Applying a score to each director...')


        #if you want to filter down the matrix to just successes:
    #     df_pop_matrix = df_pop_matrix[df_pop_matrix['success?'] == 1]
    #     df_pop_matrix.drop('success?',axis=1,inplace=True)


        #Apply the score to each writer
        matrix = pd.DataFrame((df_pop_matrix['score'].values * df_pop_matrix.iloc[:,7:].values.T).T,
                                  columns=df_pop_matrix.columns[7:], index=df_pop_matrix.index)
        df_pop_matrix = pd.concat([df_pop_matrix.iloc[:,:7],matrix],axis=1)

        #Sort by release_date and take cumsum so the score is relative to the time..
        df_pop_matrix = df_pop_matrix.sort_values('release_date',ascending=True)
        df_pop_matrix.iloc[:,7:] = df_pop_matrix.iloc[:,7:].cumsum(axis=0)
        df_pop_matrix.iloc[:,7:] = df_pop_matrix.iloc[:,7:].shift(1)
        df2 = df_pop_matrix.copy()
        df_pop_matrix.set_index('title',inplace=True)
        print('---Director-Movie matrix done! Calculating a population score for each director...')


        #Create a popularity score for each actor based on the column sums of the sparse matrix (df_pop)
        #Note: this should only be used for predict.py now
        df_dir_pop_score = pd.DataFrame(columns=['director','popularity_score'])

        i = 0
        for col in df_pop_matrix.columns[7:]:
            df_dir_pop_score.loc[i,'director'] = col
            df_dir_pop_score.loc[i,'popularity_score'] = df_pop_matrix[col].max()
            i+=1

        df_dir_pop_score.set_index('director',inplace=True)
        print('---director population dataframe done! Adding metrics to main dataframe..')
        df1 = df1.sort_values('release_date',ascending=True)
        for i, row in df1.iterrows():
            directors = row['Director']
            star_power = 0
            #take just the max popularity of all of the directors
            star_power_dir = [df_pop_matrix.loc[row['title'],director] for director in directors]
            star_power = np.mean(star_power_dir)
    #         print('total star power of {} in {}: {}'.format(row['director'],row['title'],star_power))
            df1.loc[i,'director_power'] = star_power

        print('---Complete')
        self.df_master_transform = df1
        self.df_pop_matrix_directors = df_pop_matrix
        self.director_pop_score = df_dir_pop_score

        if inplace == False:
            new_self = deepcopy(self)
            self = orig_self
            return new_self

    def last_movie_award(self, inplace=True):
        '''
        Add a metric that accounts for any awards given to the last movie that the director came out with
        '''
        if inplace == True:
            orig_self = deepcopy(self)

        df1 = self.df_master_transform.copy()

        print('---applying a score for the awards given to the director\'s last movie')
        df1 = df1[(df1['Awards'].str.contains('win',na=False)) |
                  (df1['Awards'].str.contains('won',na=False))].copy()
        df1['Awards_won'] = df1['Awards'].apply(lambda x: x.split())\
                                         .apply(lambda x: [int(s) for i, s in enumerate(x) if s.isdigit() and (x[i+1][:3] == 'win' or x[i-1].lower() == 'won')])\
                                         .apply(lambda x: sum(x))

        df1 = df1[['id', 'title', 'release_date', 'success?','Director', 'Writer', 'Actors', 'Awards_won']]
        df2 = self.df_master_transform[['id', 'title', 'release_date','success?','Director', 'Writer', 'Actors']].copy()
        df2 = pd.merge(df1, df2, on=['id', 'title','release_date', 'success?'], how='right')
        df2 = df2.drop(['Director_x','Writer_x','Actors_x'],axis=1)
        df2.rename(columns={'Director_y':'Director',
                            'Writer_y':'Writer',
                            'Actors_y':'Actors'},inplace=True)
        df2['Awards_won'].fillna(-1,inplace=True)
        df2['Awards_won'] =  df2.apply(lambda x: -1 if x['success?'] == 0 else x['Awards_won'],axis=1)
        df_pop_matrix = df2.copy()

    #Do for Director --
    #   blow out the actors into a sparse one-hot matrix:
        mlb = MultiLabelBinarizer()
        ohe = pd.DataFrame(mlb.fit_transform(df_pop_matrix['Director']),columns=mlb.classes_, index=df_pop_matrix.index)
        df_pop_matrix_dir = pd.concat([df_pop_matrix,ohe],axis=1)
        df_pop_matrix_dir.drop(['Director','Writer', 'Actors'],axis=1,inplace=True)

        print('---Director-Movie matrix initialized! Applying a score to each director...')

        matrix = pd.DataFrame((df_pop_matrix_dir['Awards_won'].values * df_pop_matrix_dir.iloc[:,5:].values.T).T,
                                  columns=df_pop_matrix_dir.columns[5:], index=df_pop_matrix_dir.index)

        df_pop_matrix_dir = pd.concat([df_pop_matrix_dir.iloc[:,:5], matrix],axis=1)

    #Do for Writers --
    #   blow out the actors into a sparse one-hot matrix:
        ohe = pd.DataFrame(mlb.fit_transform(df_pop_matrix['Writer']),columns=mlb.classes_, index=df_pop_matrix.index)
        df_pop_matrix_wr = pd.concat([df_pop_matrix,ohe],axis=1)
        df_pop_matrix_wr.drop(['Director','Writer', 'Actors'],axis=1,inplace=True)

        print('---Actors-Movie matrix initialized! Applying a score to each director...')

        matrix = pd.DataFrame((df_pop_matrix_wr['Awards_won'].values * df_pop_matrix_wr.iloc[:,5:].values.T).T,
                                  columns=df_pop_matrix_wr.columns[5:], index=df_pop_matrix_wr.index)

        df_pop_matrix_wr = pd.concat([df_pop_matrix_wr.iloc[:,:5], matrix],axis=1)

    #Do for Actors --
    #   blow out the actors into a sparse one-hot matrix:
        ohe = pd.DataFrame(mlb.fit_transform(df_pop_matrix['Actors']),columns=mlb.classes_, index=df_pop_matrix.index)
        df_pop_matrix_act = pd.concat([df_pop_matrix,ohe],axis=1)
        df_pop_matrix_act.drop(['Director','Writer', 'Actors'],axis=1,inplace=True)

        print('---Writer-Movie matrix initialized! Applying a score to each director...')

        matrix = pd.DataFrame((df_pop_matrix_act['Awards_won'].values * df_pop_matrix_act.iloc[:,5:].values.T).T,
                                  columns=df_pop_matrix_act.columns[5:], index=df_pop_matrix_act.index)

        df_pop_matrix_act = pd.concat([df_pop_matrix_act.iloc[:,:5], matrix],axis=1)

    #   #Sort by release_date..
        df_pop_matrix_dir = df_pop_matrix_dir.sort_values('release_date',ascending=True)
        df_pop_matrix_wr = df_pop_matrix_wr.sort_values('release_date',ascending=True)
        df_pop_matrix_act = df_pop_matrix_act.sort_values('release_date',ascending=True)
        df3 = self.df_master_transform.copy()
        for i, row in df3.iterrows():
            rd = row['release_date']
            directors = row['Director']
            actors = row['Actors']
            writers = row['Writer']
            _last_movie_awards = []

            for director in directors:
    #             print(director,row['title'], rd)
                try:
                    last_movie_awards = df_pop_matrix_dir[(df_pop_matrix_dir[director] != 0) & (df_pop_matrix_dir['release_date']<rd)]\
                                                 [['title','release_date','Awards_won',director]].iloc[-1:,3].values[0]
                except:
                    last_movie_awards = 0
                _last_movie_awards.append(last_movie_awards)
    #         print('--row number {}, directors last movies (max): {}'.format(i, np.max(directors_last_movie_awards)))
            df3.loc[i,'directors_last_movie_awards'] = np.max(_last_movie_awards)
            _last_movie_awards = []

            for actor in actors:
                try:
                    last_movie_awards = df_pop_matrix_act[(df_pop_matrix_act[actor] != 0) & (df_pop_matrix_act['release_date']<rd)]\
                                                 [['title','release_date','Awards_won',actor]].iloc[-1:,3].values[0]
                except:
                    last_movie_awards = 0
                _last_movie_awards.append(last_movie_awards)
            df3.loc[i,'actors_last_movie_awards'] = np.max(_last_movie_awards)
            _last_movie_awards = []

            for writer in writers:
                try:
                    last_movie_awards = df_pop_matrix_wr[(df_pop_matrix_wr[writer] != 0) & (df_pop_matrix_wr['release_date']<rd)]\
                                                 [['title','release_date','Awards_won',writer]].iloc[-1:,3].values[0]
                except:
                    last_movie_awards = 0
                _last_movie_awards.append(last_movie_awards)
            df3.loc[i,'writers_last_movie_awards'] = np.max(_last_movie_awards)
            _last_movie_awards = []
        print('---complete')
        self.df_master_transform = df3
        self.df_director_last_award = df_pop_matrix_dir
        self.df_actor_last_award = df_pop_matrix_act
        self.df_writer_last_award = df_pop_matrix_wr

        if inplace == False:
            new_self = deepcopy(self)
            self = orig_self
            return new_self

    def top_production_score(self, inplace=True):
        '''
        Makes a feature that is based off whether one of the movies'  the production companies.
        is a 'good company'. A 'good production company' is defined here as having
        over 50 titles produced and has succeeded over 60% of the time
        '''
        if inplace == True:
            orig_self = deepcopy(self)

        df1 = self.df_master_transform.copy()

        top_prod_companies = ['Universal Pictures','20th Century Fox','Paramount Pictures'
                              'Sony Pictures','Sony Pictures','Walt Disney Pictures','MCA Universal Home Video','Twentieth Century Fox Home Entertainment',
                              'Buena Vista', 'United Artists']
        df1['top_production']=df1['production_companies'].apply(lambda x: len(list(set(top_prod_companies).intersection(x))))

        self.df_master_transform = df1
        if inplace == False:
            new_self = deepcopy(self)
            self = orig_self
            return new_self

    def add_chem_factor(self,verbose=True, inplace=True):
        '''
        Add a metric that scores each film by the amount of chemistry between members of the cast
        the scoring metric is a ratio of a count of the number of successful films were worked on by exhaustive pairs
        within the crew-set (includes the director and all of the actors) and total number of crew members

        input
        -----
        verbose: bool. Default True. will print results from each movie including the amount of times
                       specific pairs worked together.
        '''
        if inplace == False:
            orig_self = deepcopy(self)

        df1 = self.df_master_transform.copy()

        df1['crew'] = None
        for i, row in df1.iterrows():
            crew = [actor for actor in row['Actors']]
            [crew.append(director) for director in row['Director']]
            crew = [x.strip(' ') for x in crew]
            df1.at[i,'crew'] = crew

        #Create an 'crew population' matrix:
        df_pop_matrix = df1[['title','crew','success?','release_date']].copy()


        #blow out the actors into a sparse one-hot matrix:
        mlb = MultiLabelBinarizer()
        ohe = pd.DataFrame(mlb.fit_transform(df_pop_matrix['crew']),columns=mlb.classes_, index=df_pop_matrix.index)
        df_pop_matrix = pd.concat([df_pop_matrix,ohe],axis=1)
        df_pop_matrix.drop('crew',axis=1,inplace=True)

        #if you want to filter down the matrix to just successes:
        df_pop_matrix = df_pop_matrix[df_pop_matrix['success?'] == 1]

        print('---crew-Movie matrix initialized! Applying a score to each crew...')

        #calculate ensemble factor
        for idx, row in df1.iterrows():
            chemCount = 0
            if verbose == True: print(row['crew'])
            for i, crew1 in enumerate(row['crew']):
                for j, crew2 in enumerate(row['crew']):
                    if j > i and crew1 != crew2:
                        count = df_pop_matrix[(df_pop_matrix['title'] != row['title']) &
                                              (df_pop_matrix[crew1] == 1) &
                                              (df_pop_matrix[crew2] == 1) &
                                              (df_pop_matrix['release_date'] < row['release_date'])].shape[0]
                        if verbose == True:
                            print ('---{} and {} have been together {} other times'.format(crew1, crew2, count))
                        chemCount += count
            if verbose == True:
                print('movie: {}, crew size: {}. chemistry count: {}'.format(row['title'], len(row['crew']), chemCount))
            df1.loc[idx,'chemistry_factor'] = chemCount
    #     df1.drop('crew',axis=1,inplace=True)

        #make the chemistry rating a ratio to the size of the crew specified
        df1['crew_size'] = df1['crew'].apply(lambda x: len(x))
        df1['chemistry_to_crew'] = df1['chemistry_factor'] / df1['crew_size']
        df1.drop('crew',axis=1,inplace=True)
        df1.drop('chemistry_factor',axis=1,inplace=True)
        df1.drop('crew_size',axis=1,inplace=True)

        print('---Wow, DONE! chemistry factor has been added!')
        self.df_master_transform = df1
        self.movie_crew_matrix = df_pop_matrix

        if inplace == False:
            new_self = deepcopy(self)
            self = orig_self
            return new_self

    def add_macro_trend(self, inplace=True):
        '''
        Add a feature to measure the industry performance by way of total avg revenue from previous year (scaled)
        '''
        if inplace == False:
            orig_self = deepcopy(self)

        df_rev = self.df_revenue_data.copy()
        df_rev['release_date'] = pd.to_datetime(df_rev['release_date'])
        df_rev.dropna(axis=0,subset=['release_date'],inplace=True)
        df_rev['Year'] = df_rev['release_date'].apply(lambda x: int(x.year))
        df_rev = df_rev.groupby('Year').mean()['revenue'].reset_index()
        df_rev.drop(index=(df_rev[df_rev['Year'] == df_rev['Year'].max()].index),inplace=True)

        s_ = np.std(df_rev['revenue'])
        u_ = np.mean(df_rev['revenue'])
        df_rev['rev_scaled'] = df_rev['revenue'].apply(lambda x: (x - u_)/s_+1)
        df_rev.set_index('Year',inplace=True)

        df1 = self.df_master_transform.copy()

        #note - if you want revenue scaled, use rev_scaled
        df1['lastYear_outlook'] = df1['release_date'].apply(lambda x: df_rev.loc[int(x.year - 1),'revenue']
                                                            if int(x.year) > 1925
                                                            else df_rev.loc[1925,'rev_scaled'])
        print('---Added industry performance metric..')

        self.df_master_transform = df1

        if inplace == False:
            new_self = deepcopy(self)
            self = orig_self
            return new_self

    def add_release_proximity(self, inplace=True):
        '''
        Add a feature to measure the amount of other movies that came out around the same time, (within a
        30 day window of the release date) which may affect the ticket sales of the movies themselves.
        '''
        if inplace == False:
            orig_self = deepcopy(self)

        df1 = self.df_master_transform.copy()
        for i,row in df1.iterrows():
            l_dateWindow = row['release_date'] - datetime.timedelta(days=15)
            u_dateWindow = row['release_date'] + datetime.timedelta(days=15)
            df1.loc[i, '30_day_proximity'] = df1[(df1['release_date'] < u_dateWindow) &
                                                 (df1['release_date'] > l_dateWindow) &
                                                 (df1['success?'] == 1)]\
                                                 .shape[0]

        print('---30 day proximity count has been added')

        self.df_master_transform = df1

        if inplace == False:
            new_self = deepcopy(self)
            self = orig_self
            return new_self

    def final_transform(self,nlp=True):
        '''
        Once significant features have been added, final_transform formats the data
        so it can be fitted to a model
        '''
        df1 = self.df_master_transform.copy()
            #genres - One hot encode list into multiple column selections:
        mlb = MultiLabelBinarizer()
    #     df1 = df1.join(pd.DataFrame(mlb.fit_transform(df1.pop('genres')),
    #                               columns=np.array(['genres_{}'.format(x) for x in mlb.classes_]),
    #                               index=df1.index))

       #Try ---
        df1.drop('genres',axis=1,inplace=True)

        #imdb_id - drop
        df1.drop('imdb_id',axis=1,inplace=True)

        #original_language - one-hot encode
        df1 = df1.join(pd.get_dummies(df1.pop('original_language'),prefix='orig_lang'))

        #overview -- drop for now.. will use for NLP later:
        df1.drop('overview',axis=1,inplace=True)

    #     #production_companies - One hot encode list into multiple column selections:
    #     df1 = df1.join(pd.DataFrame(mlb.fit_transform(df1.pop('production_companies')),
    #                               columns=np.array(['prod_comp_{}'.format(x) for x in mlb.classes_]),
    #                               index=df1.index))

        #Try--- dropping the production_companies:
        df1.drop('production_companies',axis=1,inplace=True)

        #production_countries - One hot encode list into multiple column selections:
        df1 = df1.join(pd.DataFrame(mlb.fit_transform(df1.pop('production_countries')),
                                  columns=np.array(['prod_country_{}'.format(x) for x in mlb.classes_]),
                                  index=df1.index))


        #release_date - extract the time of year (month from the date) and one-hot
        df1['release_date'] = df1['release_date'].apply(lambda x: x.to_pydatetime())\
                                                         .apply(lambda x: x.month)
        df1 = df1.join(pd.get_dummies(df1.pop('release_date'),prefix='release_month'))

        #spoken_languages - One hot encode list into multiple column selections:
    #     df1 = df1.join(pd.DataFrame(mlb.fit_transform(df1.pop('spoken_languages')),
    #                               columns=np.array(['spk_lang_{}'.format(x) for x in mlb.classes_]),
    #                               index=df1.index))

        #Try---
        df1.drop('spoken_languages',axis=1,inplace=True)

        #drop 'revenue' column due to data leakage:
        df1.drop('revenue',axis=1,inplace=True)

        #tagline -- drop for now.. will use for NLP later:
        df1.drop('tagline',axis=1,inplace=True)

        #title -- metadata. drop for now.
        df1.drop('title',axis=1,inplace=True)

        #profit -- keep for target values

        #success? -- keep for target values

        #Actors -- drop for now.. will use for NLP later:
        df1.drop('Actors',axis=1,inplace=True)

        #Country -- drop since its too similar to others
        df1.drop('Country',axis=1,inplace=True)

        #Director -- drop for now.. will use for NLP later:
        df1.drop('Director',axis=1,inplace=True)

        #Genre -- drop since its too similar to others
    #     df1.drop('Genre',axis=1,inplace=True)

        #Try ---
        df1 = df1.join(pd.DataFrame(mlb.fit_transform(df1.pop('Genre')),
                                  columns=np.array(['Genre_{}'.format(x) for x in mlb.classes_]),
                                  index=df1.index))

        #Language -- drop since its too similar to others
        df1.drop('Language',axis=1,inplace=True)

        #Try---
    #     df1 = df1.join(pd.DataFrame(mlb.fit_transform(df1.pop('Language')),
    #                           columns=np.array(['Lang_{}'.format(x) for x in mlb.classes_]),
    #                           index=df1.index))

        #Plot -- drop for now.. will use for NLP later:
        df1.drop('Plot',axis=1,inplace=True)

        #Production -- drop since its too similar to others
        df1.drop('Production',axis=1,inplace=True)

        #Try--- keeping and one-hotting the list of the Production companies:
    #     df1 = df1.join(pd.DataFrame(mlb.fit_transform(df1.pop('Production')),
    #                               columns=np.array(['prod_comp_{}'.format(x) for x in mlb.classes_]),
    #                               index=df1.index))

        #Rated - one-hot encode
        df1 = df1.join(pd.get_dummies(df1.pop('Rated'),prefix='Rated'))

        #Released -- drop since its too similar to others
        df1.drop('Released',axis=1,inplace=True)

        #Title -- metadata. drop for now.
        df1.drop('Title',axis=1,inplace=True)

        #Writer -- drop for now.. may use for NLP later:
        df1.drop('Writer',axis=1,inplace=True)

        #Year -- drop for now.. will use for macro-economic trends later:
    #     df1.drop('Year',axis=1,inplace=True)

        #Drop rating, votes and popularity -- these were only used for feature engineering
        df1.drop(['imdbRating','imdbVotes','popularity'],axis=1,inplace=True)

    #     df1.drop('chemistry_factor',axis=1,inplace=True)

        df1.dropna(axis=0,subset=['director_power'],inplace=True)
        df1.dropna(axis=0,subset = ['writer_power'],inplace=True)
        df1.dropna(axis=0,subset = ['star_power'], inplace=True)

        #Awards - drop
        df1.drop('Awards',axis=1,inplace=True)

        #id - drop
        df1.drop('id',axis=1,inplace=True)

        #nlp - drop the keywords and tokens
        if nlp == True:
         df1.drop(['Dominant_Topic','Dominant_Topic_Keywords','Tokens','plot_overview'],axis=1,inplace=True)

        self.X = df1
        self.y = self.X.pop('success?')
        self.y2 = self.X.pop('profit')
        self.X_short = self.X[['belongs_to_collection','runtime','n_prod_companies',
                            'n_prod_countries','n_spoken_languages','budget','star_power',
                            'writer_power','director_power','chemistry_to_crew','lastYear_outlook',
                            '30_day_proximity', 'Year','directors_last_movie_awards','top_production']]

    def fit(self, model_type='gb', evaluate=10, drop_features=False, **kwargs):
        '''
        Fits data to a classification model and calls evaluation

        inputs
        ------
        classifier: string. default = 'gb'. Defines which model to fit to. Options include:
                    'gb' or 'GB' = Gradient Boosting
                    'rf' or 'RF' = Random Forest

        evaluate: int or float > 0. Default = 10.
                  Determines whether a KFold will be used to cross-validate / evaluate.
                  if an integer value > 1 is passed, a kfold will be used with that number of splits/
                  if a float between 0 and 1 is passed, only a train_test_split will be used with that percent of the data.
                  if evaluate=None, no evaluation is done and model is fitted to all data.
                  Plotting is not possible if evalute=None

        drop_features: bool. default = False. Determines whether to use the shortened version of just the top features

        **kwargs: the hyperparameters to be used to instantiate the model object
        '''

        if model_type.lower() == 'gb':
            clf = GradientBoostingClassifier(**kwargs)
        elif model_type.lower() == 'rf':
            clf = RandomForestClassifier(**kwargs)
        else:
            print('Error -- must specify either rf or gb')
            return None

        if evaluate:
            self._eval_model(clf, evaluate=evaluate,drop_features=drop_features)

        if drop_features == True:
            self.trained_model = clf.fit(self.X_short,self.y)
            return self
        else:
            self.trained_model = clf.fit(self.X,self.y)
            return self

    def _eval_model(self, clf, evaluate=10, drop_features=False):

        #if K-Folding:
        if (isinstance(evaluate,int) or isinstance(evaluate,float)) and evaluate>0 and evaluate<self.X.shape[0]:
            self.evaluate = evaluate
        else:
            print('Invalid evaluate argument (must specify split or test size). Cannot evalute')
            pass

        if int(evaluate) > 1:
            print('---Fitting and evaluating the model using KFold CV\n')
            kf = KFold(n_splits=int(evaluate),shuffle=True)
            i = 0

            acc_scores = []
            auc_scores = []
            recall_scores = []
            precision_scores = []

            for train_index, test_index in kf.split(self.X.values):
                i +=1
                if drop_features:
                    X_train, X_test = self.X.values[train_index], self.X_short.values[test_index]
                else:
                    X_train, X_test = self.X.values[train_index], self.X.values[test_index]
                y_train, y_test = self.y.values[train_index], self.y.values[test_index]
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
                clf = clf.fit(X_train, y_train, sample_weight=sample_weights)
                y_pred_proba = clf.predict_proba(X_test)
                y_pred = clf.predict(X_test)

                acc_scores.append(clf.score(X_test,y_test))
                auc_scores.append(roc_auc_score(y_test,y_pred_proba[:,1]))
                recall_scores.append(recall_score(y_test,y_pred))
                precision_scores.append(precision_score(y_test,y_pred))

                print('Fold number {}. Accuracy Score: {:2.4f}. Total Acc Mean: {:2.4f}'.format(i,clf.score(X_test,y_test),np.mean(acc_scores)))
                print(' - AUC Score: {:2.4f}, AUC Mean: {:2.4f}'.format(roc_auc_score(y_test,y_pred_proba[:,1]),np.mean(auc_scores)))
                print(' - Recall: {:2.4f}, Recall Mean: {:2.4f}'.format(recall_score(y_test,y_pred), np.mean(recall_scores)))
                print(' - Precision: {:2.4f}, Precision Mean: {:2.4f}\n'.format(precision_score(y_test,y_pred),np.mean(precision_scores)))

            # print('Mean Accuracy {:2.4f}, mean AUC {:2.4f}'.format(np.mean(acc_scores),np.mean(auc_scores)))

        #For remaining evaluation, use train_test_split to get a more representative fit:
        if evaluate < 1:
            test_size = evaluate
        else:
            test_size = 0.33
        print('Fitting and evaluating the model using a 30% test-train split CV..')
        X_train, X_test, y_train, y_test = train_test_split(self.X.values, self.y.values, test_size=test_size,shuffle=True)
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        clf = clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred_proba = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        print('Accuracy Score: {:2.4f} \n \
              ROC AUC: {:2.4f} \n \
              Recall: {:2.4f} \n \
              Precision Mean: {:2.4f}'.format(clf.score(X_test,y_test),
                                              roc_auc_score(y_test,y_pred_proba[:,1]),
                                              recall_score(y_test,y_pred),
                                              precision_score(y_test,y_pred)))
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

    def plot_performance(self):
        '''
        Function for making a confusion matrix and ROC plot of the evaluation.
        Must be run after the eval_model method
        '''
        if self.evaluate == None:
            print('In order to plot performance, evaluate must be set to True during fit.')
            pass
        fig, ax = plt.subplots(figsize=(6,4))
        fpr, tpr, thresholds = roc_curve(self.y_test,self.y_pred_proba[:,1])
        ax.plot(fpr,tpr)
        ax.set_title('ROC Plot for {}'.format(type(self.trained_model).__name__))
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        # ax.legend()
        plt.show()

        class_names = np.array(['failure', 'success'])
        fig, axs = plt.subplots(1,2,figsize=(16,5))
        # Plot non-normalized confusion matrix
        axs[0] = self._plot_confusion_matrix(axs[0], classes=class_names,
                              title='Confusion matrix, without normalization')
        # Plot normalized confusion matrix
        axs[1] = self._plot_confusion_matrix(axs[1], classes=class_names, normalize=True,
                              title='Normalized confusion matrix')
        fig.tight_layout()
        plt.show()

    def _plot_confusion_matrix(self, ax, classes, normalize=False, title=None,cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(self.y_test, self.y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

    #     print(cm)

    #     fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        return ax


if __name__ == '__main__':
    #ingest data:
    df_tmbd = pd.read_csv('data/final_starting_dataset.csv',lineterminator='\n')
    df_omdb = pd.read_csv('data/omdb_data.csv')

    #Initiate / cleaning:
    model=BuildModel(df_tmbd, df_omdb)

    #Feature Engineer:
    model.transform_data()
    model.add_star_power()
    model.add_writer_power()
    model.add_director_power()
    model.last_movie_award()
    model.top_production_score()
    model.add_chem_factor(verbose=False)
    model.add_macro_trend()
    model = model.add_release_proximity(inplace=False)

    #Add NLP Topic Modeling:
    TModel = TopicModel(model.df_master_transform)
    TModel.make_lda_model()
    model.df_master_transform = TModel.perc_contribution_score()

    #Train model:
    model.final_transform(nlp=True)
    model = model.fit(evaluate=0.33,
                            learning_rate=0.05,
                            max_depth=2,
                            min_samples_leaf= 5,
                            n_estimators=500,
                            subsample= 0.8,
                            max_features=1.0)

    #Plot evaluation:
    model.plot_performance()

    #Save object:
    if 'anaconda' in sys.prefix:
        pickle.dump(model, open("data/_model.pkl", "wb"))
        pickle.dump(TModel, open("data/_topic_model.pkl", "wb"))
    else:
        pickle.dump(model, open("data/_model_conda.pkl", "wb"))
        pickle.dump(TModel, open("data/_topic_model_conda.pkl", "wb"))
