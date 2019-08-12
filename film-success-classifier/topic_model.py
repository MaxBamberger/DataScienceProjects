import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

from gensim import corpora, models
from gensim.models.phrases import Phraser
from gensim.models.wrappers import LdaMallet
from gensim.models import Phrases, CoherenceModel
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from gensim.models.wrappers import LdaMallet
from gensim.models import LdaModel

import string
import pandas as pd
import numpy as np
import re
import os
import pprint as pp
import pickle
import sys

class TopicModel:

    def __init__(self, df, min_tok_len = 1):
        '''
        Extract plot text and turn into a list of lists of tokens
        '''
        print('---Initializing Topic Modeling Object')

        df1 = df.copy()
        df1['plot_overview'] = df1.apply(lambda x: x['Plot']+' '+x['overview'],axis=1)
        plots = df1['plot_overview'].tolist()
        self.orig_df = df1

        stop_words = set(nltk.corpus.stopwords.words('english'))
        lemm_stemm = lambda tok: WordNetLemmatizer().lemmatize(tok, pos='v')

        results = []
        
        for plot in plots:
            result=[]
            
            #remove proper nouns
            tagged_sent = pos_tag(plot.split())
            noProper = [word for word,pos in tagged_sent if pos != 'NNP']
            noProper = ' '.join(noProper)
          
            for token in simple_preprocess(noProper):
                if len(token) > min_tok_len and token not in stop_words:
                    result.append(lemm_stemm(token))

            results.append(result)
      
        self.token_list = results
        
        # n_gram it up
        results = self._n_gram()
        print('  - Tokenization complete')

    def _n_gram(self, n=3):
        # Build the bigram and trigram models
        bigram = Phrases(self.token_list, min_count=5, threshold=10) # higher threshold fewer phrases.
        trigram = Phrases(bigram[self.token_list], threshold=10)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)

        if n == 3:
            self.token_list = [trigram_mod[bigram_mod[doc]] for doc in self.token_list]
        if n == 2:
            self.token_list = [bigram_mod[doc] for doc in self.token_list]

    def make_lda_model(self,num_topics=11):
        '''
        build an optimized LDA Mallet model.
        Returns coherence score for sanity checking (EDA has revealed the target coherence to be ~0.39)
        '''
        print('  - Building LDA Model model with {} topics'.format(num_topics))

        dictionary = corpora.Dictionary(self.token_list)
        corpus = [dictionary.doc2bow(text) for text in self.token_list]

        #set up mallet path
        # os.environ.update({'MALLET_HOME':r'anaconda3/lib/python3.7/site-packages/mallet-2.0.8/'})
        # mallet_path = '/anaconda3/lib/python3.7/site-packages/mallet-2.0.8/bin/mallet' # update this path
        #
        # #Make Model:
        # ldamallet = LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=20)
        #Get Coherence Score:
        coherence_score = CoherenceModel(model=ldamodel, texts=self.token_list, dictionary=dictionary, coherence='c_v').get_coherence()
        # model_topics = optimal_model.show_topics(formatted=False)

        # print topics
        pp.pprint(ldamodel.print_topics(num_words=6))
        print("  - Num Topics: {}. Coherence Value of: {:2.3f}".format(num_topics, coherence_score))

        self.all_topics = ldamodel.print_topics(num_words=6)
        self.ldamodel = ldamodel
        self.corpus = corpus
        self.dictionary = dictionary
        self.coherence_score = coherence_score

    def perc_contribution_score(self):

        print('  - Applying a contribution score to each film from original dataset')
        # Init output
        first_topic_df = pd.DataFrame()
        second_topic_df = pd.DataFrame()
        perc_contribution_df = pd.DataFrame()

    # Get main topic in each document
        for i, row in enumerate(self.ldamodel[self.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            perc_contribution = {}
            for j, (topic_num, prop_topic) in enumerate(row):

                if j == 0:  # => dominant topic
                    wp = self.ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    first_topic_df = first_topic_df.append(pd.Series([int(topic_num), topic_keywords]), ignore_index=True)

                perc_contribution['perc_contribution_topic'+str(topic_num)] = round(prop_topic,4)

            perc_contribution_df = perc_contribution_df.append(pd.DataFrame(pd.Series(perc_contribution)).transpose(),ignore_index=True)

        first_topic_df.columns = ['Dominant_Topic', 'Dominant_Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.DataFrame(pd.Series(self.token_list),columns=['Tokens'])
        print(contents.shape[0],first_topic_df.shape[0],perc_contribution_df.shape[0])
        perc_contribution_df.fillna(0,inplace=True)
        sent_topics_df = pd.concat([first_topic_df,perc_contribution_df, contents], axis=1)
        print('  - Application completed.. merging with original dataframe')
        df_withTopics = pd.concat([self.orig_df,sent_topics_df],axis=1)
        self.df_withTopics = df_withTopics

        print('  - Topic Model complete!')
        return df_withTopics

if __name__ == '__main__':
    #For testing and debugging:
    try:
        from build_model2 import BuildModel
        df_tmbd = pd.read_csv('data/final_starting_dataset.csv',lineterminator='\n')
        df_omdb = pd.read_csv('data/omdb_data.csv')
        model=BuildModel(df_tmbd, df_omdb)
        model.transform_data()
    except ImportError:
        unpickle = open("data/_model.pkl","rb")
        model = pickle.load(unpickle)
    df = model.df_master_transform.copy()

    #build model:
    TModel = TopicModel(df)
    TModel.make_lda_model()
    df_withtopics = TModel.perc_contribution_score()

    print(df_withtopics[['title', 'Dominant_Topic', 'Dominant_Topic_Keywords','Plot', 'success?']].head())

    #save model:
    if 'anaconda3' in sys.prefix:
        pickle.dump(TModel, open("data/_topic_model_conda-test.pkl", "wb"))
    else:
        pickle.dump(TModel, open("data/_topic_model-test.pkl", "wb"))
