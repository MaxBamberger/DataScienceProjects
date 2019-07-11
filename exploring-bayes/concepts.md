


1. What is the prior associated with choosing any one die?

>P(die(ANY)) = 0.20

2. What is the likelihood function? You should assume that the die are all fair.

>P( roll | die) 

3. Say I roll an 8. After one bayesian update, what is the probability that I chose each of the dice?

>P( die 4 | roll 8) = 0
>P( die 6 | roll 8) = 0
>P( die 8 | roll 8) = ((0.125)(0.20))/(0 + 0 + 0.125*(0.20) + 0.083*(0.20) + 0.05*(0.20)) =  0.025/0.0516 = 0.4845    
>P( die 12 | roll 8) = ((0.083)(0.20))/(0 + 0 + 0.125*(0.20) + 0.083*(0.20) + 0.05*(0.20)) = 0.01666/0.0516 = 0.3217    
>P( die 20 | roll 8) = ((0.05)(0.20))/(0 + 0 + 0.125*(0.20) + 0.083*(0.20) + 0.05*(0.20)) =  0.025/0.0516 = 0.194

4. Comment on the difference in the posteriors if I had rolled the die 50 times instead of 1.

>The posterior is affected by the conditional probability of the joint probability of each of the 50 rolls that happen

5. Which one of these two sets of data gives you a more certain posterior and why? [1, 1, 1, 3, 1, 2] or [10, 10, 10, 10, 8, 8]

>The second dataset. Only two dice can possibly have sides 10 or greater, whereas each one of the dice has all of the sides in the first dataset.

6. Say that I modify my prior by my belief that bigger dice are more likely to be drawn from the box. This is my prior distribution:

 4-sided die: 8%
 6-sided die: 12%
 8-sided die: 16%
 12-sided die: 24%
 20-sided die: 40%
 
What are my posteriors for each die after rolling the 8?

>P( die 4 | roll 8) = 0
>P( die 6 | roll 8) = 0
>P( die 8 | roll 8) = ((0.125)(0.16))/(0 + 0 + 0.125*(0.16) + 0.083*(0.24) + 0.05*(0.40)) =  0.02/0.05992 = 0.3337  
>P( die 12 | roll 8) = ((0.083)(0.24))/(0 + 0 + 0.125*(0.16) + 0.083*(0.24) + 0.05*(0.40)) = 0.01992/0.05992 = 0.332   
>P( die 20 | roll 8) = ((0.05)(0.40))/(0 + 0 + 0.125*(0.16) + 0.083*(0.24) + 0.05*(0.40)) =  0.02/0.05992 = 3337

Which die do we think is most likely? Is this different than what you got with the previous prior?

>8-sided, 12-sided and 20 sided die are all basicially equally likely now.

7. Say you keep the same prior and you roll the die 50 times and get values 1-8 every time. What would you expect of the posterior? How different do you think it would be if you'd used the uniform prior?

>I would expect the 20, 12 and 8-sided dice to all have similar probaility, with the 20-sided die just marginally more probable. If you used a uniform prior like before then the 8-sided die would be more likely.