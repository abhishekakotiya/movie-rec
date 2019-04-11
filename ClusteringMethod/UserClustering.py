#!/bin/python
from movielens import *
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

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

test = np.zeros((n_users, n_items))
for r in rating_test:
    test[r.user_id - 1][r.item_id - 1] = r.rating

# The utility matrix stores the rating for each user-item pair in the matrix form.
# The movielens data is indexed starting from 1 
utility = np.zeros((n_users, n_items))
for r in rating:
    utility[r.user_id-1][r.item_id-1] = r.rating

print utility

clusters = 7
kmean_cluster = KMeans(n_clusters = clusters)
kmean_cluster.fit(utility)
labels = kmean_cluster.predict(utility)

# Creating a dictionary of {cluster, [user]} {key, value} pair
user_cluster = {}
for i in range(clusters):
    user_cluster[i] = []

user_no = 1
for l in labels:
    user_cluster[l].append(user_no)
    user_no += 1

# Calculating average rating of every movie for every cluster
cluster_ratings = []
for i in range(n_items):
    average = np.zeros(clusters)
    tmp = []
    for i in range(clusters):
        tmp.append([])
    for j in range(n_users):
        if utility[j][i]:
            tmp[labels[j]].append(utility[j][i])
    for k in range(clusters):
        if len(tmp[k]):
            average[k] = np.mean(tmp[k])
        else:
            average[k] = 0
    cluster_ratings.append(average)
cluster_ratings = np.array(cluster_ratings)
cluster_ratings = cluster_ratings.transpose()

# Finds the Pearson Correlation Similarity Measure between two users
def pcs(x, y):
    num = 0
    den1 = 0
    den2 = 0
    A = cluster_ratings[x]
    B = cluster_ratings[y]
    num = sum((a - user[x].avg_r) * (b - user[y].avg_r) for a, b in zip(A, B) if a > 0 and b > 0)
    den1 = sum((a - user[x].avg_r) ** 2 for a in A if a > 0)
    den2 = sum((b - user[y].avg_r) ** 2 for b in B if b > 0)
    den = (den1 ** 0.5) * (den2 ** 0.5)
    if den == 0:
        return 0
    else:
        return num / den

# User_cluster-user_cluster similarity
# Finding PCSM value for all user_cluster-user_cluster pairs
pcs_matrix = np.zeros((clusters, clusters))
for i in range(clusters):
	for j in range(clusters):
		if i != j:
			pcs_matrix[i][j] = pcs(i, j)

print pcs_matrix

# Find average rating for each user and store it in the user's object
for key, value in user_cluster.items():
    """x = cluster_ratings[key]
    rated = np.nonzero(x)
    if rated:
        average = np.mean(x[rated])
    else:
        average = 0
    for v in value:
        user[v-1].avg_r = average 
    """
    for i in range(n_users):
        rated = np.nonzero(utility[i])
        n = len(rated[0])
        if n != 0:
            user[i].avg_r = np.mean(utility[i][rated])
        else:
            user[i].avg_r = 0.

# Normalize the user ratings
def norm():
	normalize = np.zeros((clusters, n_items))
	for i in range(clusters):
		for j in range(0, n_items):
			if cluster_ratings[i][j]:
				normalize[i][j] = cluster_ratings[i][j] - user[i].avg_r
			else:
				normalize[i][j] = float('Inf')
	return normalize            

# Guesses the ratings that user with id, user_id, might give to item with id, i_id.
# We will consider the top_n similar users to do this.
def guess(user_id, i_id, top_n):
    similarity = []
    id_label = labels[user_id-1]
    for i in range(clusters):
    	if i != id_label:
    		similarity.append(pcs_matrix[id_label][i])
    temp = norm()
    temp = np.delete(temp, id_label, 0)
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

utility_copy = np.copy(cluster_ratings)
for i in range(clusters):
    for j in range(n_items):
        if utility_copy[i][j] == 0:
            utility_copy[i][j] = guess(i, j, 4)

print utility_copy

# Predict ratings for u.test and find the mean squared error
y_true = []
y_pred = []
for i in range(0, n_users):
    for j in range(0, n_items):
        if test[i][j] > 0:
            y_true.append(test[i][j])
            y_pred.append(utility_copy[labels[i]][j])

print "Mean Squared Error: %f" % mean_squared_error(y_true, y_pred)**0.5
