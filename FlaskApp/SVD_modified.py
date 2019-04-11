"""
Computing utility matrix and similar_movies file to be given as input to the actual recommender program.
No need to run this file if data_dumps are already created.

"""

#!/bin/python
from movielens import *
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr

import numpy as np
import csv
import time
import math
import time
import pickle

"""

Part - 1: Begin

Data Loading 

"""

# Store data in arrays
user = []
item = []
rating = []
rating_test = []

# Load the movie lens dataset into arrays
d = Dataset()
d.load_users("data/u.user", user)
d.load_items("data/u.item", item)
d.load_ratings("data/u.base", rating)
d.load_ratings("data/u.test", rating_test)

n_users = len(user)
n_items = len(item)

# The utility matrix stores the rating for each user-item pair in matrix form.
# The loaded movielens data has indexing starting from 1 
utility = np.zeros((n_users, n_items))
for r in rating:
    utility[r.user_id-1][r.item_id-1] = r.rating

"""

Part - 1: End

"""


"""

Required functions

"""


# Finds the average rating for each user
for i in range(n_users):
    rated = np.nonzero(utility[i])
    n = len(rated[0])
    if n != 0:
        user[i].avg_r = np.mean(utility[i][rated])
    else:
        user[i].avg_r = 0.

def averagerating():
    avg = 0
    n = 0
    for i in range(len(rating)):
        avg += rating[i].rating
        n += 1
    return float(avg/n)

test = np.zeros((n_users, n_items))
for r in rating_test:
    test[r.user_id - 1][r.item_id - 1] = r.rating


# predict rating
def predict(i, j):
    # calculating dot product of i and j
    r = sum([i[k]*j[k] for k in range(len(i))])
    if r > 5:
        r = 5
    elif r < 1:
        r = 1
    return r


# Modified version of SVD to fill the sparse utility matrix

def SVDModified(R, U, V, rank, maxepochs=30, lrate=0.035, regularizer=0.01, minimprov = 0.001):
    oldtrainerr = 1000000.0
    for k in xrange(rank):
        for epoch in range(maxepochs):
            sse = 0.0
            n = 0
            for i in range(len(rating)):
                crating = rating[i]
                err = crating.rating - predict(U[crating.user_id-1], V[crating.item_id-1])
                sse += err**2
                n += 1

                uTemp = U[crating.user_id-1][k]
                vTemp = V[crating.item_id-1][k]

                U[crating.user_id-1][k] += lrate * (err*vTemp - regularizer*uTemp)
                V[crating.item_id-1][k] += lrate * (err*uTemp - regularizer*vTemp)

            trainerr = sse/n

            if abs(oldtrainerr-trainerr) < minimprov:
                break
            oldtrainerr = trainerr
    return U, V



"""

Part - 2: Begin

Initialization

""" 

init_time = time.time()
R = np.array(utility)

nU = len(R)
vU = len(R[0])
rank = 30

avg = averagerating()
initval = math.sqrt(avg/rank)
        
# U matrix
U = [[initval]*rank for i in range(n_users)]
# V matrix - easier to store and compute than V^T
V = [[initval]*rank for i in range(n_items)]

U = [[math.sqrt(avg/rank)]*rank for i in range(n_users)]
V = [[math.sqrt(avg/rank)]*rank for i in range(n_items)]

#U = np.random.rand(nU,rank)
#V = np.random.rand(vU,rank)

"""

Part - 2: End

"""


"""

Part - 3: Begin

Filling the sparse utility matrix and finding RMSE value 

"""

P, Q = SVDModified(R, U, V, rank)

print "time: ", time.time()-init_time

M = np.dot(np.array(P), np.array(Q).T)
M.dump("./data_dumps/utility.dat")

print M

# Predict ratings for u.test and find the mean squared error
y_true = []
y_pred = []
for i in range(0, n_users):
    for j in range(0, n_items):
        if test[i][j] > 0:
            y_true.append(test[i][j])
            y_pred.append(M[i][j])

print "Mean Squared Error: %f" % mean_squared_error(y_true, y_pred)

"""

Part - 3: End

"""


""" 

Part - 4: Begin

calculating item similarity to tackle cold start and/or recommend similar movies

"""



utility = np.array(M).T

# storing item-item similarity values based on pearson correlation similarity
pcs_matrix = np.zeros((n_items, n_items))

for i in range(n_items):
    for j in range(n_items):
        if i != j:
            pcs_matrix[i][j] = pearsonr(utility[i], utility[j])[0]

pcs_matrix.dump("./data_dumps/item_similarity.dat")

"""

Part - 4: End

"""


"""

Part - 5: Begin

Finding top - 5 similar movies for every movie

"""

similar_movies = {}
item_similarity = pcs_matrix

movie_id = range(1, 1683)

for i in range(1682):
    similar_movies[i+1] = [x for (x,y) in sorted(zip(movie_id, item_similarity[i]), key=lambda pair:pair[1], reverse=True)]
    similar_movies[i+1] = similar_movies[i+1][:5]

f = open('./data_dumps/similar_movies', 'w')
pickle.dump(similar_movies, f)
f.close()

# f2 = open('similar_movies', 'r')
# s = pickle.load(f2)
# f2.close()

# print s

"""

Part - 5: End

"""

print "Execution successful"