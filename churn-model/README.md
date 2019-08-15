## Churn Prediction

Many businesses these days are subscription-based. Even some businesses that aren't charging customers periodically still rely on the continuous engagement of their customers which is ineffect following a similar model.

All subscription-modeled enterprises fear the same threat: churn. Losing a subscribed customer is often much more costly than offering deep promotions to keep them.. as gaining new subscribers is not always so easy. Thus knowing which customers are in danger of churning is vital to the success of their business.

In this case study I demonstrate how to build a predictive model that can forecast which customers may be in danger of 'churning' using customer data from  a ride-sharing company in San Francisco.  Since the data is sourced from a real company, I cannot share the dataset here on Github.

`pipeline_and_model.py` is my main script  which calls several functions in `helper_functions.py` to clean the data and derive latent features. I consider any customer that has not used the app in the last 6 months to be 'churned' It then trains a classifier which is used to make predictions on my test data.
