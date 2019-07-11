
from bayes import Bayes

def likelihood_func(data,key):
    key = float(key)
    data = float(data)
    faces = list(range(int(key)+1))[1:]
    if data in faces:
        return 1/key
    else: return 0

uniform_prior = {4:0.2, 6:0.2, 8:0.2, 12:0.2, 20:0.2}
unbalanced_prior = {4:0.08, 6:0.12, 8:0.16, 12:0.24, 20:0.4}

d=[8,2,1,2,5,8,2,4,3,7,6,5,1,6,2,5,8,8,5,3,4,2,4,3,8,
 8,7,8,8,8,5,5,1,3,8,7,8,5,2,5,1,4,1,2,1,3,1,3,1,5]

set1 = [1, 1, 1, 3, 1, 2]
set2 = [10, 10, 10, 10, 8, 8]

print('What are the posteriors if we started with the uniform prior?')
bayes_uniform = Bayes(uniform_prior.copy(),likelihood_func=likelihood_func)
bayes_uniform.update(8)
bayes_uniform.print_distribution()

print('What are the posteriors if we started with the unbalanced prior?')
bayes_unbalanced = Bayes(unbalanced_prior.copy(),likelihood_func=likelihood_func)
bayes_unbalanced.update(8)
bayes_unbalanced.print_distribution()

print('How different were these two posteriors (the uniform from the unbalanced)?')
for k, v in bayes_unbalanced.posterior.items():
    print("{} : {}".format(k, bayes_uniform.posterior[k] - bayes_unbalanced.posterior[k]))


print('\nApplying this set of data to the function:\n',d)
for i in d[1:]:
    bayes_uniform.update(i)
    bayes_unbalanced.update(i)

print('What are the posteriors if we started with the uniform prior?')
bayes_uniform.print_distribution()
print('What are the posteriors if we started with the unbalanced prior?')
bayes_unbalanced.print_distribution()

print('How different were these two posteriors (the uniform from the unbalanced)?')
for k, v in bayes_unbalanced.posterior.items():
    print("{} : {}".format(k, bayes_uniform.posterior[k] - bayes_unbalanced.posterior[k]))

print('\nWith the uniform prior, which of these two sets of data leads to a more certain posterior?', set1, 'or', set2)
bayes_uniform = Bayes(uniform_prior.copy(),likelihood_func=likelihood_func)
for i in set1:
    bayes_uniform.update(i)
print('\nSet1:')
bayes_uniform.print_distribution()

bayes_uniform = Bayes(uniform_prior.copy(),likelihood_func=likelihood_func)
for i in set2:
    bayes_uniform.update(i)
print('\nSet2:')
bayes_uniform.print_distribution()
print('\nSet 2 leads to a more certain posterior!')
