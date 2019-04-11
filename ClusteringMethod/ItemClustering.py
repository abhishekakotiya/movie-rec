#!/bin/python
from movielens import *
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

import math
import numpy as np

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

# The utility matrix stores the rating for each user-item pair in the matrix form.
# The movielens data is indexed starting from 1 
utility = np.zeros((n_users, n_items))
for r in rating:
    utility[r.user_id-1][r.item_id-1] = r.rating

# Finds the average rating for each user and stores it in the user's object
"""for i in range(n_users):
    rated = np.nonzero(utility[i])
    n = len(rated[0])
    if n != 0:
        user[i].avg_r = np.mean(utility[i][rated])
    else:
        user[i].avg_r = 0.
"""

test = np.zeros((n_users, n_items))
for r in rating_test:
    test[r.user_id - 1][r.item_id - 1] = r.rating

print utility

#storing only the 0 and 1 genre values in a new array
genre = []
for movie in item:
	genre.append([movie.unknown, movie.action, movie.adventure, movie.animation, 
		movie.childrens, movie.comedy, movie.crime, movie.documentary, movie.drama, 
		movie.fantasy, movie.film_noir, movie.horror, movie.musical, movie.mystery, 
		movie.romance, movie.sci_fi, movie.thriller, movie.war, movie.western])

genre = np.array(genre)
# Perform KMeans clustering using scikit-learn
kmean_cluster = KMeans(n_clusters = 10)
kmean_cluster.fit(genre)
labels = kmean_cluster.predict(genre)

# Stores the average rating by each user for each cluster
user_clustered = []
for i in range(0, n_users):
	average = np.zeros(10)
	tmp = []
	for k in range(0, 10):
		tmp.append([])
	for j in range(0, n_items):
		if utility[i][j] != 0:
			tmp[labels[j]].append(utility[i][j])
	for k in range(0, 10):
		if len(tmp[k]) != 0:
			average[k] = np.mean(tmp[k])
		else:
			average[k] = 0
	user_clustered.append(average)

user_clustered = np.array(user_clustered)

# Find average rating for each user according to the new clustered ratings and store it in the user's object
for i in range(0, n_users):
	x = user_clustered[i]
	user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)

# Finds the Pearson Correlation Similarity Measure between two users
def pcs(x, y):
    num = 0
    den1 = 0
    den2 = 0
    A = user_clustered[x - 1]
    B = user_clustered[y - 1]
    num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r) for a, b in zip(A, B) if a > 0 and b > 0)
    den1 = sum((a - user[x - 1].avg_r) ** 2 for a in A if a > 0)
    den2 = sum((b - user[y - 1].avg_r) ** 2 for b in B if b > 0)
    den = (den1 ** 0.5) * (den2 ** 0.5)
    if den == 0:
        return 0
    else:
        return num / den

# User-user similarity
# Finding PCSM value for all user-user pairs
pcs_matrix = np.zeros((n_users, n_users))
for i in range(0, n_users):
	for j in range(0, n_users):
		if i != j:
			pcs_matrix[i][j] = pcs(i + 1,j + 1)

print pcs_matrix

# Normalize the user ratings
def norm():
	normalize = np.zeros((n_users, 10))
	for i in range(0, n_users):
		for j in range(0, 10):
			if user_clustered[i][j] != 0:
				normalize[i][j] = user_clustered[i][j] - user[i].avg_r
			else:
				normalize[i][j] = float('Inf')
	return normalize            

# Guesses the ratings that user with id, user_id, might give to item with id, i_id.
# We will consider the top_n similar users to do this.
def guess(user_id, i_id, top_n):
    similarity = []
    for i in range(0, n_users):
    	if i+1 != user_id:
    		similarity.append(pcs_matrix[user_id-1][i])
    temp = norm()
    temp = np.delete(temp, user_id-1, 0)
    top = [x for (y,x) in sorted(zip(similarity,temp), key=lambda pair: pair[0], reverse=True)]
    s = 0
    c = 0
    for i in range(0, top_n):
    	if top[i][i_id-1] != float('Inf'):
    		s += top[i][i_id-1]
    		c += 1
    g = user[user_id-1].avg_r if c == 0 else s/float(c) + user[user_id-1].avg_r
    if g < 1.0:
        return 1.0
    elif g > 5.0:
        return 5.0
    else:
        return g

utility_copy = np.copy(user_clustered)
for i in range(0, n_users):
    for j in range(0, 10):
        if utility_copy[i][j] == 0:
            utility_copy[i][j] = guess(i+1, j+1, 150)

print utility_copy

# Predict ratings for u.test and find the mean squared error
y_true = []
y_pred = []
for i in range(0, n_users):
    for j in range(0, n_items):
        if test[i][j] > 0:
            y_true.append(test[i][j])
            y_pred.append(utility_copy[i][labels[j]])

print "Mean Squared Error: %f" % mean_squared_error(y_true, y_pred)
