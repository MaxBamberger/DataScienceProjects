from collections import Counter
from flask import Flask, request, session
import pickle
from build_model2 import BuildModel
from predict import Predict
import ast
import sys
import gensim
from gensim import corpora, models
from gensim.models.phrases import Phraser
from gensim.models.wrappers import LdaMallet
from gensim.models import Phrases, CoherenceModel
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from gensim.models.wrappers import LdaMallet
from sklearn.preprocessing import MultiLabelBinarizer
import pprint as pp

app = Flask(__name__)
# sess = Session()
# app.secret_key = "super secret key"

session_output = 0

if 1 == 2:
    print('opening conda pickles')
    unpickle = open("data/_model_conda.pkl","rb")
    model = pickle.load(unpickle)
    unpickle = open("data/_topic_model_conda.pkl","rb")
    tModel = pickle.load(unpickle)
else:
    unpickle = open("data/_model.pkl","rb")
    model = pickle.load(unpickle)
    unpickle = open("data/_topic_model.pkl","rb")
    tModel = pickle.load(unpickle)

@app.route('/')
def input():
    with open('form.html', 'r') as file:
        page = file.read()
    return page

@app.route('/model_predict', methods=['POST'] )
def predict():
    title = request.form['element_19']
    belongs_to_collection = bool(True if int(request.form['element_2']) == 1 else False)
    budget = int(float(request.form['element_6_1']+'.'+request.form['element_6_2']))
    genre = str(request.form['element_3']).lower()
    orig_lang = str(request.form['element_5']).lower()
    spoken_lang = str(request.form['element_4']).lower()
    prd_comp = str(request.form['element_8']).lower()
    prd_countries = request.form['element_7'].lower()
    rel_date = str(request.form['element_9_1']+'-'+request.form['element_9_2']+'-'+request.form['element_9_3'])
    runtime = int(request.form['element_10'])
    thirty_day_proximity = int(request.form['element_11'])
    rated = str(request.form['element_18'])
    director = str(request.form['element_12']).lower()
    writers = str(request.form['element_13']).lower()
    actors = str(request.form['element_14']).lower()
    plot = str(request.form['element_15'])
    tagline = str(request.form['element_16'])
    last_year_outlook = int(float(request.form['element_17_1']+'.'+request.form['element_17_2']))

    test_data = {'Title': title,
                  'belongs_to_collection':belongs_to_collection,
                  'budget': budget,
                  'genres': genre,
                  'original_language': orig_lang,
                  'spoken_languages': spoken_lang,
                  'production_companies': prd_comp,
                  'production_countries': prd_countries,
                  'release_date': rel_date,
                  'runtime': runtime,
                  '30_day_proximity': thirty_day_proximity,
                  'rated': rated,
                  'director': director,
                  'writer': writers,
                  'actors': actors,
                  'plot': plot,
                  'tagline' : tagline,
                  'lastYear_outlook' : last_year_outlook}

    # test_data = ast.literal_eval(test_data)
    print('Prediction in progress!\n')
    pp.pprint(test_data)
    print('')
    pred = Predict(test_data)
    output = []
    output.append(pred.initial_transform())
    output.append(pred.add_star_power(actor_pop_scores=model.actor_pop_scores))
    output.append(pred.add_writer_power(writer_pop_scores = model.writer_pop_scores))
    output.append(pred.add_director_power(director_pop_score = model.director_pop_score))
    output.append(pred.last_movie_award(director_awards=model.df_director_last_award,
                          actor_awards=model.df_actor_last_award,
                          writer_awards=model.df_writer_last_award))
    output.append(pred.top_production_score())
    output.append(pred.add_chem_factor(movie_crew_matrix = model.movie_crew_matrix))

    
    output.append(pred.topic_model(dictionary=tModel.dictionary, trained_token_list=tModel.token_list,lda_model=tModel.ldamodel))
    
    output.append('\n These are all of the possible latent topics: \n')
    
    topics = tModel.ldamodel.print_topics(num_words=6)
    keywords = [str(i)+'. '+x[1] for i, x in enumerate(topics)]
    keywords = '\n'.join(keywords)
    output.append(keywords)

    pred.finalize_for_model(model=model)
    pred.predict_one(trained_model=model.trained_model)

    global session_output
    session_output = output
    return '''<h1 style="background: #FFFFFF; text-shadow: 2px 2px 0 #bcbcbc, 4px 4px 0 #9c9c9c; color: #000000;">Thank you for waiting. Here are your results!</h1>
                <div style="background: #FFFFFF; text-shadow: 2px 2px 0 #bcbcbc, 4px 4px 0 #9c9c9c; color: #000000;">&nbsp;</div>
                <p><a href="https://github.com/MaxBamberger/DataScienceProjects/tree/master/film-success-classifier">Learn more about the model that's being applied</a></p>
                <p>&nbsp;</p>
                <table style="border-color: black; float: left; width: 603px; margin-left: 30px;" border="black" cellpadding="1">
                <tbody style="padding-left: 30px;">
                <tr style="height: 18px; padding-left: 30px;">
                <td style="height: 18px; width: 390px;"><em>&nbsp; Probability of Success:</em></td>
                <td style="height: 18px; width: 201px;">&nbsp; {}</td>
                </tr>
                <tr style="height: 18px; padding-left: 30px;">
                <td style="height: 18px; width: 390px;"><em>&nbsp; Outcome:</em></td>
                <td style="height: 18px; width: 201px;">&nbsp; {}</td>
                </tr>
                <tr style="height: 18px; padding-left: 30px;">
                <td style="height: 18px; width: 390px;"><em>&nbsp; Advice:</em></td>
                <td style="height: 18px; width: 201px;">&nbsp; {}</td>
                </tr>
                <tr style="height: 20.8438px; padding-left: 30px;">
                <td style="height: 20.8438px; width: 390px;"><em>&nbsp; Dominant Topics in your movie:</em></td>
                <td style="height: 20.8438px; width: 201px;">&nbsp; {}</td>
                </tr>
                </tbody>
                </table>
                <p style="padding-left: 30px;">&nbsp;</p>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p><a href="http://3.224.227.110:8080/explain_prediction">Why Did I get These Results?</a></p>
                <p>&nbsp;</p>
                <p><a href="https://github.com/MaxBamberger/DataScienceProjects/tree/master/film-success-classifier">Learn more about the model that's being applied</a></p>
                <p>&nbsp;</p>
                <p><a href="http://3.224.227.110:8080">Try Another Movie Prediction</a></p>
             '''.format(pred.pred_perc, pred.pred_label, pred.advice, pred.keywords)

@app.route('/explain_prediction', methods=['GET'] )
def explain():
    if type(session_output) == int:
        return '''Error: you need to make a prediction to get Results<br>
                    <p><a href="http://3.224.227.110:8080">Try Another Movie Prediction</a></p>'''
    output = '<br>'.join(session_output)
    output = output.replace('\n','<br>')
    return output+'''
                <p>&nbsp;</p>
                <p><a href="https://github.com/MaxBamberger/DataScienceProjects/tree/master/film-success-classifier">Learn more about the model that's being applied</a></p>
                <p>&nbsp;</p>
                <p><a href="http://3.224.227.110:8080">Try Another Movie Prediction</a></p>'''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
