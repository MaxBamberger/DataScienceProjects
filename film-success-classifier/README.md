# Fame or Flop? 
## Predicting the Success of a Film Using Engineered Features, Natural Language Processing (LDA) and Machine Learning

Hollywood is a ruthless business. Achieving success in industry this means everything one could want in life: fame, fortune, legacy and adoration. But the stakes are high: 
70% of movies that are made have a negative ROI.. Meaning they lose money for the original production studios. Many producers, writers and actors struggle their entire life in vain for just a small glimmer of the limelight.

**What if you could know with more certainty if a movie has a real shot at success before your money is sunk?**

Lots of people complain that "movies these days are so formuliac".. Well, there may be some truth to that statement. I've set out to create a machine learning classification system that -- when trained with over 7000+ films throughout history -- gets to the bottom of what this formula is.

## Try it out! 
You can try [making a prediction](http://54.159.9.172:8080/) on a future movie with the web-app version of my model 
(Put together for demonstration purposes only. I never claimed to be a great web-developer :))

### About the model itself:
The main success/failure and predicted probability is performed with a Gradient Boosting algorithm, however textual data such as the plot synopsis, tagline etc. is fed to a Latent Drichlet Allocation for Topic Modeling. Typically the best Coherence value for training data of this size is found with just 20 topics. These topics (and each movie's % contribution to them) are fed back into the Boosting classifier as new features. The General pipeline is as follows:
 - New features are engineering 
 - Main classifier algorithm: GradientBoosting
 - Plot Synopsis text: Latent Drichlet Allocation (Topic Modeling)


### Model scoring:

 - My confusion Matrix:
![image](images/cm.png)

### Feauture importance:

### Where does the data come from?
 - Data is ingested through two APIs:
    - asdfasfas
    - asdfasdf
 - I've created some handy tools for requesting the APIs and ingesting the data (more to come)
