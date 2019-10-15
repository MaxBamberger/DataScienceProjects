# Abstract
Many complicated natural language processing (NLP) techniques and complex deep learning models exist today for the purpose of learning and extracting human ‘sentiment’ from a body of text. Entire Python libraries like ‘spaCy’ have been built for tagging, tokenizing and properly mapping language into vectors that a machine learning or deep learning model can train on. Most commonly used NLP libraries cover the languages of the developed countries (English, German, French, Spanish etc.), but what if your body of text is in a language like Roman-Urdu? In this paper we’ll discuss Naïve Bayes, a very simple yet powerfully accurate classifier that can quickly add over 30% accuracy over random classification, even on very lightly processed text. We’ll discuss how to process and tokenize the text, Term Frequency-Inverse Document Frequency (TF-IDF) – a common method of vectorizing the text into numerical values – and how to deal with data imbalance. Because our data set does come with its limitations (as does the Naïve Bayes algorithm) we’ll briefly discuss the theory on how to combat those limitations and improve results through deep-learning techniques like Long Short-Term Memory (LSTM) Recurrent Neural Nets. 


# Business Justification
Sentiment analysis is important for any B2C business whose model is concerned with text-based customer engagement, customer-service, or online reviews/feedback. Even businesses who are not in direct contact with text-based data from their customers may be concerned with how their product is received in customer reviews or how their brand is being performing on a public or social media forum. How the masses ‘feel’ emotionally about your level of service or what you’re selling can sometimes means the difference between success and failure – especially if the market you’re in is competitive. Tracking sentiment is easy enough for a human to do intuitively but training a computer to do it offers better cost savings through automation and more efficient surveillance, buying the business more time to react to changes in their public perception.
In this case, the business is a multinational corporation seeking to automatically identify the sentiment amongst its Urdu-speaking customer base. Their goal is to train a classification model to learn sentiment with a specific interest in being able to detect the ‘Negative’ labeled classes. Since we’ve been given data that is already tagged with ‘Positive’, ‘Negative’, ‘Neutral’ labels, this problem will be solved by via training a supervised learning model.

# The Dataset
The data consists of Urdu-transliterated sentences gathered from various sources including social media and reviews of products on ecommerce sites. Each row ideally consists of a single sentence and has a corresponding sentiment attached to it that is one of three values: ‘Negative’, ‘Positive’ or ‘Neutral’. There are more than 20,000 sentences that have been manually tagged, and some of the data is inconsistent.


# Methodology
Most ML/DL problems involving NLP can be represented by a series of these general steps called a data pipeline.






In our case, the steps are simplified and less extensive due to the limitations of the dataset but we can still follow this framework all the same. 

## Ingest and Tokenize
We start by ingesting our dataset and doing some preliminary cleaning. Looking at the data, the first thing we notice is a mislabeled record and some null text values. 

Since there aren’t that many missing values, we can drop the full row of any null values. We can easily fix the misspellings.

Next we process the text into meaningful tokens by removing stop words, proper nouns, capitalization, special characters. Stop words are words that are so commonly used when stringing together a sentence, they don’t carry much meaning in terms of the overall message (in English these are prepositional words like ‘the’, ‘at’, ‘a’, ‘which’, ‘with’ etc.). For finding common stop words in Roman-Urdu to pull out as well as some common proper nouns, I used a verified list of words on Github: (https://github.com/Smat26/Roman-Urdu-Dataset).  Lastly we create bi and tri-gram representations for common 2 and 3-word phrases.  We can do this algorithmically using Gensim’s ‘phraser’ functions. 



Overall the goal here is to boil each document of text down to just their essential, standardized ‘bag of words’ representations. Other tools we might employ in further processing the text is a word mapping technique called lemmatization (and a similar technique: stemmazation) whereby we transform our words into their root or parent tense. A number of algorithms exist for lemmatization and stemmatization in English whereby the mapping looks like this: 

	Am, are, is --> be
	difference, different, differential --> differ

Unfortunately, no such autonomous technique exists for Roman-Urdu that I’m aware of, without fluency in the language so we’ll skip that for now.

Vectorization
Once our corpus is fully processed into a list of lists of tokens the next step is to vectorize our documents into numeric values so that our classifier can understand and run computation. In this case we’re going to use a sparse TF-IDF matrix with each word in the overall vocabulary as a column and each document as a row (depending upon how we tweak the model the shape of the matrix is around [18106 x 31046]). The value at each element of the matrix is the 
term frequency * log inverse of the document frequency. Thus, it increases number of times a word appears in a document and is offset by the number of documents the word appears in the overall corpus. Scikit-Learn comes with a great function for doing this:


We can tweak these hyperparameters carefully as they have a direct effect on our classifier’s ability to predict sentiment in the next step. The most important hyperparameter I found useful for optimizing this dataset was max_df: this specifies the maximum document frequency of a given word. If exceeded, the word is omitted.  

# Naïve Bayes Classifier
Finally, the main engine that’s powering our predictions is a Naïve Bayes classifier: a simple yet effective method used in many common text classification use cases like spam filtering. The math behind how it works is rooted in Bayes theorem:
Essentially, the classifier makes a (rather naïve) assumption that every word independently contributes to the overall probability of class. In this case, x is a vector transformed by the words in each document and c is our sentiment class ‘Negative’, ‘Neutral’ or ‘Positive’. To predict a class for given a document, we simply calculate the posterior probabilities for each class c and see which class has the highest. For example calculating the negative sentiment probability for a given document with n words:


P\left(negative\middle|\vec{x}\right)\ \propto P\left({word}_1\ \right|\ negative)\ \times\ldots\times P\left({word}_n\ \right|\ negative)\ \times P(negative)


Where the prior P(negative) is the observed class probability amongst the training data. To estimate P(wi|c) we conceptually use the observed word’s frequency:

\frac{N_{ji}}{N_j}=\frac{total\ count\ of\ word\ w_i\ across\ all\ documents\ of\ class\ c}{total\ count\ of\ all\ words\ across\ all\ the\ documents}

This is represented for us by the TF-IDF vectorization.

We also employ a Laplacing smoothing constant \alpha to avoid calculating 0 when we run into a new word that’s not found in the training data a P(c | wi )

P\left(w_i\middle| c\right)=\frac{\alpha+D_{ij}}{\alpha p+N_{ij}}

An alpha = ~0.25 in our case gives us the highest accuracy. Along with the TF-IDF Vectorizer parameters, this can be tuned with an exhaustive grid search.

clf = MultinomialNB(alpha=0.25)

Cross Validation
To evaluate our data we make a number of Training and Test-set splits in our data using a Stratified KFold function that shuffles and evenly distributes the classes amongst the training and test sets. At each split we fit and transform the document (X_train) values into our TF-IDF vector then fit those values to the classifier using the target classes (y_train) as our supervised targets. We then separately transform the hold-out document set (X_test) into our fitted TF-IDF vector and use it as input for making predictions on our pre-fitted classifier. An evaluation metric is calculated for the split and the process is repeated for the next split. Since the classifier doesn’t take very long to train and a higher number of splits means more data to train on, a relatively high number of splits such as 20 is appropriate here. This will roughly keep our training data to about 20000 documents and test data to about 1000.


Evaluation… what metrics do we care about?
Since our data has three target classes that are evenly distributed, the best metric to look at is accuracy. Very simply, accuracy is:

			Accuracy=\frac{values\ correctly\ predicted}{total\ predictions\ made} 

This is a harsh metric since you require that each label set be correctly predicted for every sample. With the right parameters were able to achieve an accuracy of ~65%, representing a more than 30% boost over randomly guessing a class label. 
Accuracy is not everything though.  Revisiting the problem statement: there is special interest in being able to accurately detect negative sentiment. Examining our confusion matrix:

We see that only 46% of negative sentiment labels (-1) are predicted correctly, with 44% of them getting misclassified as neutral sentiment. One way to mitigate this (albeit a little ‘hacky’) is to look at the calculated probabilities for each prediction and insert some bias. The prediction always goes to the class that has greatest probability, but we can manipulate our thresholds so that wherever we have a ‘close call’ between a neutral and negative probability we favor negative. For example for a given prediction \hat{y}:


\hat{y}\ =\ [0.3242, 0.3620, 0.3138]

If we manipulate our thresholds so that the neutral value has to surpass the negative probability by 0.05 when it’s the largest probability, we may be able to build a classification system that is more sensitive to finding true negative classes. In this example since the differences in probability between the negative and neutral class so close (less than our 0.05 threshold), give this prediction to the negative class.

There are other metrics that are better suited for classification evaluation, such as Recall, Precision and F1 score, but those apply only to a binary classification problem. Which brings us to our next point…


Resampling and Mitigating Class Imbalance 
Missing over half the negative sentiments is really not acceptable; and since it is really just negative sentiment we’re hoping to detect, what if we changed our problem to a binary classification? This will enable us to use some additional classification evaluation metrics:

	Recall or Sensitivity or TPR (True Positive Rate): Number of items correctly identified as positive out of total true positives: \left(TP\right)/\left(TP+FN\right) Intuitively this is our model’s ability to find all of the positive classes
	Precision: Number of items correctly identified as positive out of total items identified as positive (TP)/\left(TP+FP\right). Intuitively this is our models ability to avoid false positives
	F1 Score: It is a harmonic mean of precision and recall given by
 \left(2\times P r e c i s i o n\times R e c a l l\right)/\left(Precision+Recall\right)
	ROC AUC: Area under curve of sensitivity (TPR) vs. FPR (1-specificity)


There are a few scenarios we can explore where we manipulate our dataset to make the problem binary…

Alternative Scenario 1: Drop all the Neutrals

 

Dropping the neutral values on the onset of the problem will help us to train on only the extreme sentiments ‘positive’ and ‘negative’. Since ‘negative’ classes are what were after, we’re going to map our negative values to 1. Thus a ‘true positive’ means correctly classifying a negative sentiment, and a ‘false positive’ means misclassifying a truly positive sentiment as a negative sentiment. Though it may sound confusing, the reason we do this is to orient the Precision score and Recall score to be a measurement of how well we can detect negative sentiment.

Since the data is well-balanced and clear of ambiguous ‘neutral’ data, our model performs much better


Average Accuracy: 0.7768
 -- Average recall score:  0.7102
 -- Average precision score:  0.7914
 -- Average f1 score:  0.7486
 -- Average roc_auc score:  0.8636

However, these results are a bit misleading. The real data we'd like to test this model with may have neutral values that could get misclassified. 

Alternative Scenario 2: Convert Neutrals and Positives to the Same Class
In this scenario we naturally consider neutrals and positives to be part of the same class, since we’re only really concerned with negatives. We’ll encode 1 for the Negative class and 0 for the Positive, but we’ve just created a new problem: class imbalance! Class imbalances, when very severe, will cause our classifier to bias the over-weighted side and our accuracy score to become misleading.
  
Feeding just the imbalanced dataset to our classifier we find that we have an F1 and Recall score of around 0.25 and 0.15 respectively. That’s a pretty poor evaluation but exactly what you might expect when your positive class (which is really the negative sentiment) is in the minority. So how do you fight data imbalance? A common method is to resample by under-representing the majority class in your training set. An even more effective method is to over-sample by generating synthetic data using an algorithm called SMOTE (Synthetic Minority Over-sampling Technique). 

SMOTE uses a technique similar to k-nearest neighbors that randomly generates new artificial values in the surrounding ‘neighbor radius’ of points in vector space of the minority class. There are a number of parameters which you can feed to a grid-search to fine tune.



We set our sampling strategy to 1.0 to create perfectly balanced classes. Another parameter we can tweak is k_neighbors which dictates the number of points in the minority class used to create a boundary / radius used to define the space with which the synthetic points can lie.

To avoid data leakage, we do our SMOTE over-sampling ONLY on the training data once it has been transformed into a TF-IDF matrix. An effective kNN relies on scaled data, especially when dimensionality is high, since it employs Euclidian distance as a way of measuring similarity between points. Since the data being sampled has a huge number of dimensions (in the realm of 26000 features based on the vocabulary) it’s a good idea to scale the data as well.  

The results we get are a very slight downtick in accuracy but a significant boost in Recall Score and F1 Scores over the same classification without using a SMOTE synthesizer:





In conclusion, a binary classification with SMOTE gives us our best results and sets us up most realistically for detecting negative sentiment on text data in the wild. To productionize the solution, we can train a model on all of our data and pickle (store) the model as an object to import for making real-time predictions.


Addressing Limitations 
Thinking ahead towards improving the accuracy, we’d first look to the address some limitations of the data and my own lack of familiarity with the language of Urdu:

Inconsistent/Inaccurate labeling. Since some of the data labels have been inconsistently and manually labeled – it’s subject to interpretation. Human intuition can fall short, especially if trying to interpret the mood of someone through text. I would opt to try and augment this process through a topic modeling solution. Many algorithms exist for finding the latent topics of a document such as Latent Drichlet Allocation. Through a topic modeler we can boil down each of our documents into k number of topics that may clue us into the sentiment (for instance, we might learn that that topic x has to do with war which carries a 90% chance of being associated to negative sentiment, while topic y which has to do with family which has a good chance of being positive). A solution like this allows us to revisit sentiment values from both a programmatic and intuitive point of view.

Better NLP functionality. I was fortunate enough to find a text file online for Roman-Urdu stop words and some common proper nouns, but the full potential of NLP is far from being realized. If time were no obstacle, we could spend more of it building out a full list of proper nouns, learning parts of speech and even building a lemmatization or stemmatization function for mapping words to their origin. This would also call for much more requisite knowledge of the language of Urdu.

Limitations of Naïve Bayes. We call our classifier ‘naïve’ because it’s based on the assumption that every word contributes independently to the sentiment. In reality, words are seldom independent of each other. We depend on context clues which is why a more robust deep learning solution would be better long term. 

…Further Optimize Using Deep-Learning
Beyond improving the dataset and our technique for pre-processing our text, perhaps the most effective ROI for our time spent would be to use instead a deep learning solution. Long Short-
Term Memory (LSTM) neural nets are a type of Recurrent Neural Net (RNN) that performs extraordinarily well with sequence data such as time series data or text data. Like all other neural nets, output values are sent from one layer of artificial neurons to another and the output at each layer is computed by some nonlinear function with weights that adjust as learning proceeds. However with an LSTM (and RNNs in general), there is recurrence so that some information can 



persist while new data enters as input. Where LSTM differs from other traditional RNNs is the complexity of the repeating module within each neuron. 



The repeating module in an LSTM

With this added structure, LSTMs are able to recall and connect to previous information with the current input, whereas a plain RNN could only handle very ‘recent’ information. 

In a way this mimics how we read and process language naturally – we make connections based on contextual dependencies both long and short-term. If we were to predict the next word in any given sentence, it would depend not only on words in that sentence, but the subject matter, the predicate of the previous sentence and maybe even declaratives made at the very beginning of the paragraph.
To apply an LSTM for this type of problem, we’d have to first change the way we vectorize words. Rather than a sparse matrix of our entire vocabulary, we’d want to set a specific boundary on column dimensions and represent each document as a sequence of word/tokens (maintaining order), padding zeroes when the sentence falls short of the column length and truncating sentences when they run over. Rather than a value based on term frequency each word in the vocab would instead be converted to a unique integer. 

A general architecture and approach to training an LSTM sentiment analyzer is described in the figure on the previous page

Following this approach, building out the neural net with TFLearn (a modular library built on top of Tensorflow) might look like this: 
 

With our input dimensions of the embedding layer matching the output dimensions of the input layer (usually all or an important segment of our vocabulary). The resulting word embeddings are fed to the LSTM layer which we can apply a modest dropout to avoid overfitting. The fully connected layer gets fed the output of the LSTM layer. Adding a fully connected layer is a way of learning non-linear combinations of the feature vectors from all of the neurons in the previous step (with the activation ‘softmax’ giving us probabilities). The final ‘regression’ layer optimizes a given loss function and specifies how fast we want our network to train.

While this code was applied to our data set, I refer to it here purely as a theoretical example. Some additional work remains to hyper-tune the model and improve accuracy. Unfortunately training these types of deep learning models takes quite a bit of time!

As a conscientious next step towards improving our model, I would strongly advice a move towards developing a sentiment analysis solution that utilizes LSTM. Once properly trained an LSTM can perform at the head of the class – in some cases learning to predict sentiments with over a 95% accuracy!


 



















# References
	N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, SMOTE: Synthetic Minority Over-sampling Technique, Jun 1 2002 Vol 16. https://jair.org/index.php/jair/article/view/10302
	Hussain Ghulama, Feng Zenga, Wenjia Lib, Yutong Xiaoa Deep Learning-Based Sentiment Analysis for Roman-Urdu. Feb 6, 2019.
	Aniruddha Choudhury, “Sentiment Classifications with Natural Language Processing on LSTM”. Jan 12, 2019.  https://blog.usejournal.com/sentiment-classification-with-natural-language-processing-on-lstm-4dc0497c1f19 
	Samarth Agrawal, “Sentiment Analysis using LSTM”. Feb 18, 2019 https://towardsdatascience.com/algorithms-for-text-classification-part-1-naive-bayes-3ff1d116fdd8
	https://github.com/Smat26/Roman-Urdu-Dataset
	https://keras.io/examples/imdb_lstm/ 
	https://en.wikipedia.org/wiki/Word_embedding
	https://en.wikipedia.org/wiki/Naive_Bayes_classifier
	https://www.sciencedirect.com/science/article/pii/S1877050919302200
	https://colah.github.io/posts/2015-08-Understanding-LSTMs/
