import pickle
import sys
import os
from build_model2 import BuildModel
from topic_model import TopicModel
import pandas as pd
import re
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

import gensim
from gensim import corpora, models
from gensim.models.phrases import Phraser
from gensim.models.wrappers import LdaMallet
from gensim.models import Phrases, CoherenceModel
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from gensim.models.wrappers import LdaMallet
from sklearn.preprocessing import MultiLabelBinarizer


class Predict:
    '''
    Predict Class.
    Creates object with methods for transforming one test-record (one-row pandas dataframe into a
    format for making a prediction against a train classifier. Called by the web-app.py for predicting user inputted data.

    Type:  object

    inputs
    ------
    test_data: dict

    methods
    -------
    initial_transform()
    add_star_power()
    add_writer_power()
    add_director_power()
    last_movie_award()
    top_production_score()
    add_chem_factor()
    topic_model()
    finalize_for_model(trained_model)
    predict_one(trained_model, thresh=0.5)

    '''
    def __init__(self, test_data):
        self.keywords = None
        self.test_data = pd.Series(test_data).to_frame().transpose()

    def initial_transform(self):
        '''
        Main function for cleaning the raw dataset.
        '''
        df1 = self.test_data.copy()

        #Make 'belongs_to_collection a true/false
        df1['belongs_to_collection'] =  df1['belongs_to_collection'].apply(lambda x : 1 if x == True else 0)

        #Isolate the genre data into lists
        df1['genres'] =  df1['genres'].apply(lambda x : [a.strip() for a in x.split(',')])

        #Isolate the production_companies data into lists and note the # of production companies
        df1['production_companies'] =  df1['production_companies'].apply(lambda x : [a.strip() for a in x.split(',')])
        df1['n_prod_companies'] = df1['production_companies'].apply(lambda x : len(x))

        #Isolate the production_countries data into lists and note the # of production countries
        df1['production_countries'] =  df1['production_countries'].apply(lambda x : [a.strip() for a in x.split(',')])
        df1['n_prod_countries'] = df1['production_countries'].apply(lambda x : len(list(x)))

        #Isolate the spoken_languages data into lists and note the # of spoken_languages
        df1['spoken_languages'] = df1['spoken_languages'].apply(lambda x : [a.strip() for a in x.split(',')])
        df1['n_spoken_languages'] = df1['spoken_languages'].apply(lambda x : len(list(x)))

        #Transform to the right dtypes
        df1['original_language'] = df1['original_language'].astype(str)
        df1['plot'] = df1['plot'].astype(str)
        df1['release_date'] = pd.to_datetime(df1['release_date'])
        df1['runtime'] = df1['runtime'].astype(int)
        df1['tagline'] = df1['tagline'].astype(str)
        df1['actors'] = df1['actors'].astype(str)
        df1['actors'] = df1['actors'].apply(lambda x : [a.strip() for a in x.split(',')])
        df1['director'] = df1['director'].astype(str)
        df1['director'] = df1['director'].apply(lambda x: [a.strip() for a in x.split(',')])
        df1['rated'] = df1['rated'].astype(str)
        df1['Title'] = df1['Title'].astype(str)

        regex = re.compile(".*?\((.*?)\)")
        df1['writer'] = df1['writer'].astype(str)
        df1['writer'] = df1['writer'].apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x).split(','))\
                                     .apply(lambda x: [w.strip(' ') for w in x])\
                                     .apply(lambda x: list(dict.fromkeys(x)))

        df1['Year'] = df1['release_date'].apply(lambda x: x.to_pydatetime())\
                                                         .apply(lambda x: x.year)

        print('--Test dataframe initilialized with movie metadata!')
        self.test_data = df1

    def add_star_power(self, actor_pop_scores):
        '''
        Adds a metric to the transformed data that measures a movies' 'star-power' score
        based on previous results
        '''
        df1=self.test_data.copy()
        actors = df1.loc[0,'actors']
        star_power_actors = []
        for actor in actors:
            try:
                star_power_actors.append(actor_pop_scores.loc[actor,'popularity_score'])
            except KeyError as e:
                star_power_actors.append(0)
        df1.loc[0,'star_power']=np.median(star_power_actors)

        output = '--Actors: {}\'s median popularity: {:2.4f} added to dataframe!'.format(actors, np.median(star_power_actors))
        print(output)
        self.test_data = df1
        return output

    def add_writer_power(self,writer_pop_scores ):
        '''
        Adds a metric to the transformed data that measures the success of the movie's writer / staff
        based on previous results
        '''
        df1=self.test_data.copy()
        writers = df1.loc[0,'writer']
        star_power_writers = []
        for writer in writers:
            try:
                star_power_writers.append(writer_pop_scores.loc[writer,'popularity_score'])
            except KeyError as e:
                star_power_writers.append(0)
        df1.loc[0,'writer_power']=np.max(star_power_writers)

        output = '--Writer(s): {}\'s popularity: {:2.4f} added to dataframe!'.format(writers, np.max(star_power_writers))
        print(output)
        self.test_data = df1
        return output

    def add_director_power(self,director_pop_score):

        df1=self.test_data.copy()

        directors = df1.loc[0,'director']
        star_power_dir = []
        for director in directors:
            try:
                star_power_dir.append(director_pop_score.loc[director,'popularity_score'])
            except KeyError as e:
                star_power_dir.append(0)
        df1.loc[0,'director_power']=np.max(star_power_dir)

        output = '--Director(s): {}\'s popularity: {:2.4f} added to dataframe!'.format(directors, np.max(star_power_dir))
        print(output)
        self.test_data = df1
        return output

    def last_movie_award(self, director_awards, actor_awards, writer_awards):
        '''
        Add a metric that accounts for any awards given to the last movie that the director came out with
        '''
        df1=self.test_data.copy()

        directors = df1.loc[0,'director']
        actors = df1.loc[0,'actors']
        writers = df1.loc[0,'writer']

        last_movie_awards = []
        for director in directors:
            try:
    #             print(director)
                last_movie_awards.append(director_awards[director_awards[director] != 0]\
                                        [['title','release_date','Awards_won',director]].iloc[-1:,3].values[0])
            except KeyError as e:
                last_movie_awards.append(0)

        #value will be -1 if their last movie was a flop.. will want to correct for this..
        last_movie_awards = [0 if x < 0 else last_movie_awards[i] for i, x in enumerate(last_movie_awards)]
        df1.loc[0,'directors_last_movie_awards'] = np.max(last_movie_awards)

        last_movie_awards = []
        for actor in actors:
            try:
                last_movie_awards.append(actor_awards[(actor_awards[actor] != 0)]\
                                        [['title','release_date','Awards_won',actor]].iloc[-1:,3].values[0])
            except KeyError as e:
                last_movie_awards.append(0)

        last_movie_awards = [0 if x < 0 else last_movie_awards[i] for i, x in enumerate(last_movie_awards)]
        df1.loc[0,'actors_last_movie_awards'] = np.max(last_movie_awards)

        last_movie_awards = []
        for writer in writers:
            try:
                last_movie_awards.append(writer_awards[(writer_awards[writer] != 0)]\
                                        [['title','release_date','Awards_won',writer]].iloc[-1:,3].values[0])
            except KeyError as e:
                last_movie_awards.append(0)

        last_movie_awards = [0 if x < 0 else last_movie_awards[i] for i, x in enumerate(last_movie_awards)]

        df1.loc[0,'writers_last_movie_awards'] = np.max(last_movie_awards)

        output1 = '--Added {} awards from Director(s): {} last movie'.format(df1.loc[0,'directors_last_movie_awards'], directors)
        output2 = '--Added {} awards. This is the maximum from each of the actors(s): {} last movies'.format(df1.loc[0,'actors_last_movie_awards'], actors)
        output3 = '--Added {} awards. This is the maximum from each of the writers(s): {} last movies'.format(df1.loc[0,'writers_last_movie_awards'] , writers)
        output = output1+'\n'+output2+'\n'+output3
        print(output)
        self.test_data = df1
        return output

    def top_production_score(self):
        '''
        Makes a feature that is based off the movies' production companies having status as a 'top production house'
        A 'top production company' is defined here as having over 50 titles produced and has succeeded over 60% of the time
        
        input
        -----
        self
        
        output
        ------
        output: str. print messages for the web-app explanation page 
        '''
        df1=self.test_data.copy()

        top_prod_companies = ['Universal Pictures','20th Century Fox','Paramount Pictures'
                              'Sony Pictures','Sony Pictures','Walt Disney Pictures',
                              'MCA Universal Home Video','Twentieth Century Fox Home Entertainment',
                              'Buena Vista', 'United Artists']

        df1['top_production']=df1['production_companies'].apply(lambda x: len(list(set(top_prod_companies).intersection(set(x)))))

        output = '--Total # of production houses that are top tier: {} '.format(df1.loc[0,'top_production'])
        print(output)
        self.test_data = df1
        return output

    def add_chem_factor(self, movie_crew_matrix):
        '''
        Adds a metric that scores the film by the amount of 'chemistry' between members of the cast.
        The scoring metric is a ratio of a count of the number of successful films were worked on by exhaustive pairs
        within the crew-set (includes the director and all of the actors) and total number of crew members
                
        input
        -----
        self
        
        output
        ------
        output: str. print messages for the web-app explanation page 
        '''
        df1=self.test_data.copy()
        masterOutput = ''
        #initialize a crew column for the df:
        df1['crew'] = None
        for i, row in df1.iterrows():
            crew = [actor for actor in row['actors']]
            [crew.append(director) for director in row['director']]
            crew = [x.strip(' ') for x in crew]
            df1.at[i,'crew'] = crew

        #find the combinations of times each pair of crew-members worked together:
        for idx, row in df1.iterrows():
            chemCount = 0
            output = '--Measuring chemistry between crew members:'
            print(output)
            masterOutput = masterOutput + '\n' + output
            output = ', '.join(row['crew'])
            print(output)
            masterOutput = masterOutput + '\n' + output

            for i, crew1 in enumerate(row['crew']):
                for j, crew2 in enumerate(row['crew']):
                    if j > i and crew1 != crew2:
                        try:
                            count = movie_crew_matrix[(movie_crew_matrix['title'] != row['Title']) &
                                                      (movie_crew_matrix[crew1] == 1) &
                                                      (movie_crew_matrix[crew2] == 1) &
                                                      (movie_crew_matrix['release_date'] < row['release_date'])].shape[0]
                        except KeyError as e:
                            output =  '{} not found'.format(e.args)
                            print(output)
                            masterOutput = masterOutput + '\n' + output
                            count = 0
                        output = '  ---{} and {} have been together {} other times'.format(crew1, crew2, count)
                        print(output)
                        masterOutput = masterOutput + '\n' + output
                        chemCount += count
            output = '--movie: {}, crew size: {}. chemistry count: {}'.format(row['Title'], len(row['crew']), chemCount)
            print(output)
            masterOutput = masterOutput + '\n' + output
            df1.loc[idx,'chemistry_factor'] = chemCount
    #     df1.drop('crew',axis=1,inplace=True)

        #make the chemistry rating a ratio to the size of the crew specified
        df1['crew_size'] = df1['crew'].apply(lambda x: len(x))
        df1['chemistry_to_crew'] = df1['chemistry_factor'] / df1['crew_size']
        df1.drop('crew',axis=1,inplace=True)
        df1.drop('chemistry_factor',axis=1,inplace=True)
        df1.drop('crew_size',axis=1,inplace=True)

        self.test_data = df1
        return masterOutput

    def topic_model(self, dictionary, trained_token_list, lda_model):
        '''
        Processes the plot and tagline into and adds percentage contribution to each of the 
        topics in the lda_model -- produced from topic_model.py.
        
        input
        -----
        dictionary: Gensim Dictionary object. From corpora.Dictionary(trained_token_list) in topic_model.py. pickled object
        trained_token_list: list of lists of processed tokens. Created from topic_model.py pickled object
        lda_model: Gensim ldamodel object created from topic_model.py model pickled object
        
        
        output
        ------
        output: str. print messages for the web-app explanation page 
        '''
        masterOutput = ''
        output = '--Calculating plot text\'s contribution into Topics Modeled..'
        print(output)
        masterOutput = masterOutput + '\n' +  output
        df1=self.test_data.copy()
        plot = df1.loc[0,'plot'] + " " + df1.loc[0,'tagline']

        #pre-process into tokens and n-grams
        tokens = self._preprocess(plot)
        tokens = self._n_gram(tokens, trained_token_list)

        #get perc contribution to tModel's topics
        new_dict = corpora.Dictionary(tokens)
        new_corpus = [dictionary.doc2bow(text) for text in tokens]
        topic_scores = sorted(lda_model[new_corpus][0], key=lambda x: (x[1]), reverse=True)

        #add to dataframe
        first_topic_df = pd.DataFrame()
        perc_contribution_df = pd.DataFrame()
        perc_contribution = {}

        for j, (topic_num, prop_topic) in enumerate(topic_scores):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                dom_topic_keywords = ", ".join([word for word, prop in wp])
                first_topic_df = first_topic_df.append(pd.Series([int(topic_num), dom_topic_keywords]), ignore_index=True)
                first_topic = int(topic_num)
            if j == 1:  # => 2nds most dominant topic
                wp = lda_model.show_topic(topic_num)
                second_topic_keywords = ", ".join([word for word, prop in wp])
                second_topic = int(topic_num)

            perc_contribution['perc_contribution_topic'+str(topic_num)] = round(prop_topic,4)

        perc_contribution_df = perc_contribution_df.append(pd.DataFrame(pd.Series(perc_contribution)).transpose(),ignore_index=True)
        first_topic_df.columns = ['Dominant_Topic', 'Dominant_Topic_Keywords']
        perc_contribution_df.fillna(0,inplace=True)
        sent_topics_df = pd.concat([first_topic_df,perc_contribution_df], axis=1)

        df1 = df1.join(sent_topics_df)
        self.test_data = df1
        self.keywords = dom_topic_keywords
        output = '--Topic Modeling Complete! \n--Dominant Topic: '+dom_topic_keywords+" (Topic #"+str(first_topic)+")"
        print(output)
        masterOutput = masterOutput + '\n' +  output
        output = '--Second Most Dominant Topic: '+second_topic_keywords+" (Topic #"+str(second_topic)+")"
        print(output)
        masterOutput = masterOutput + '\n' +  output
        return masterOutput


    def _preprocess(self, text, min_tok_len = 1):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        lemm_stemm = lambda tok: WordNetLemmatizer().lemmatize(tok, pos='v')

        result=[]
        
        #remove proper nouns
        tagged_sent = pos_tag(text.split())
        noProper = [word for word,pos in tagged_sent if pos != 'NNP']
        noProper = ' '.join(noProper)
        
        for token in simple_preprocess(noProper):
            if len(token) > min_tok_len and token not in stop_words:
                result.append(lemm_stemm(token))

        # Build the bigram and trigram models
        bigram = Phrases(result, min_count=5, threshold=10) # higher threshold fewer phrases.
        trigram = Phrases(bigram[result], threshold=10)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)

        result = trigram_mod[bigram_mod[result]]

        return [result]

    def _n_gram(self,tokens, trained_token_list, n=3):
        # Build the bigram and trigram models
        bigram = Phrases(trained_token_list, min_count=5, threshold=10) # higher threshold fewer phrases.
        trigram = Phrases(bigram[trained_token_list], threshold=10)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)

        if n == 3:
            return [trigram_mod[bigram_mod[tok]] for tok in tokens]
        if n == 2:
            return [bigram_mod[tok] for doc in tokens]

    def finalize_for_model(self,model):
        '''
        Format the test dataframe into the same format as the trained format so it can be used to predict
        
        input
        -----
        model: sklearn pickled object trained model from build_model2.py
        
        output
        ------
        output: str. print messages for the web-app explanation page 
        '''
        df1=self.test_data.copy()


        #one hot what you need to one hot..
        df1 = df1.join(pd.get_dummies(df1.pop('original_language'),prefix='orig_lang'))

        mlb = MultiLabelBinarizer()
        df1 = df1.join(pd.DataFrame(mlb.fit_transform(df1.pop('genres')),
                                  columns=np.array(['Genre_{}'.format(x) for x in mlb.classes_]),
                                  index=df1.index))

        df1 = df1.join(pd.DataFrame(mlb.fit_transform(df1.pop('production_countries')),
                                  columns=np.array(['prod_country_{}'.format(x) for x in mlb.classes_]),
                                  index=df1.index))

        df1['release_date'] = df1['release_date'].apply(lambda x: x.to_pydatetime())\
                                                         .apply(lambda x: x.month)

        df1 = df1.join(pd.get_dummies(df1.pop('release_date'),prefix='release_month'))

        df1 = df1.join(pd.get_dummies(df1.pop('rated'),prefix='Rated'))

        #Add blank columns to match training data
        for col in model.X.columns:
            if col not in list(df1.columns):
                df1[col] = 0


        #Cut down columns to match training data and match order
        df1 = df1[list(model.X.columns)]

        self.X_test = df1
        print('--Data Transformation is complete! Predicting outcome..\n')

    def predict_one(self, trained_model, thresh=0.5):
        '''
        Makes a prediction from on the test data using the trained model object in build_model2.py 
        
        input
        -----
        model: sklearn pickled object trained model from build_model2.py
        
        output
        ------
        output: str. print messages for the web-app explanation page 
        '''
        classes = [thresh, ((1-thresh)/4)+thresh,((1-thresh)/4)*2+thresh, ((1-thresh)/4)*3+thresh]
        pred_proba = trained_model.predict_proba(self.X_test)[0,1]
        pred = trained_model.predict(self.X_test)
        percentage = str(round((pred_proba * 100),2))+'%'

        if pred_proba > classes[0] and pred_proba < classes[1]:
            label = 'Ambiguous Success'
            advice = 'Probability is very close to threshold for success/failure. More testing is required.'

        elif pred_proba > classes[1] and pred_proba < classes[2]:
            label = 'Moderate Success'
            advice = 'Moderate probability of success. More testing strongly recommended.'

        elif pred_proba > classes[2] and pred_proba < classes[3]:
            label = 'Success'
            advice = 'A solid probability of success, though more testing is recommended.'

        elif pred_proba > classes[3]:
            label = 'Very Strong Success'
            advice = 'Very Strong probability of success'

        else:
            label = 'Flop!'
            advice = 'Probability of success was below the threshold.. failure is likely.'

        self.pred_perc = percentage
        self.pred_label = label
        self.advice = advice
        print(percentage)
        print(label)
        print(advice)
        print('Dominant Topic: ', self.keywords)

if __name__ == '__main__':
    if 'anaconda3' in sys.prefix:
        unpickle = open("data/_model_conda.pkl","rb")
        model = pickle.load(unpickle)
        unpickle = open("data/_topic_model_conda.pkl","rb")
        tModel = pickle.load(unpickle)
        print('unpickled the condas')
    else:
        unpickle = open("data/_model.pkl","rb")
        model = pickle.load(unpickle)
        unpickle = open("data/_topic_model.pkl","rb")
        tModel = pickle.load(unpickle)

    test_data = {'Title': 'Game Story 2020',
                  'belongs_to_collection': False,
                  'budget': 10,
                  'genres': 'Action, Adventure, Mystery',
                  'original_language': 'en',
                  'spoken_languages': 'en, es, fr',
                  'production_companies': '20th Century Fox, Buena Vista',
                  'production_countries': 'US',
                  'release_date': '2020-05-16',
                  'runtime': 92,
                  '30_day_proximity': 5,
                  'rated': 'R',
                  'director': 'Steven Spielberg',
                  'writer': 'Aaron Sorkin',
                  'actors': 'Tom Hanks, Gary Oldman, Jennifer Connelly, Jeff Bridges, Ian McKellen',
                  'plot': 'Years after a rebellion spurred by a stolen bride to be and the blind ambitions of a mad King, Robert of the house Baratheon (Mark Addy) sits on the much desired Iron Throne. In the mythical land of Westeros, nine noble families fight for every inch of control and every drop of power. The King\'s Hand, Jon Arryn (Sir John Standing), is dead. And Robert seeks out his only other ally in all of Westeros, his childhood friend Lord Eddard "Ned" Stark. The solemn and honorable Warden of the North is tasked to depart his frozen sanctuary and join the King in the capital of King\'s Landing to help the now overweight and drunk Robert rule. However, a letter in the dead of night informs "Ned" that the former Hand was murdered, and that Robert will be next. So noble Ned goes against his better desires in an attempt to save his friend and the kingdoms. But political intrigue, plots, murders, and sexual desires lead to a secret that could tear the Seven Kingdoms apart. And soon Eddard will find out what happens when you play the Game of Thrones.',
                  'tagline' : 'You are never too old for an adventure with your old friends..',
                  'lastYear_outlook' : 15000000}

    pred = Predict(test_data)
    pred.initial_transform()
    pred.add_star_power(actor_pop_scores=model.actor_pop_scores)
    pred.add_writer_power(writer_pop_scores = model.writer_pop_scores)
    pred.add_director_power(director_pop_score = model.director_pop_score)
    pred.last_movie_award(director_awards=model.df_director_last_award,
                          actor_awards=model.df_actor_last_award,
                          writer_awards=model.df_writer_last_award)
    pred.top_production_score()
    pred.add_chem_factor(movie_crew_matrix = model.movie_crew_matrix)

    if 0 == 1:
        pred.topic_model(dictionary=tModel.dictionary, trained_token_list=tModel.token_list, lda_model=tModel.ldamodel)
    else:
        ldamodel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(tModel.ldamodel)
        pred.topic_model(dictionary=tModel.dictionary, trained_token_list=tModel.token_list, lda_model=ldamodel)

    pred.finalize_for_model(model=model)
    pred.predict_one(trained_model=model.trained_model)
