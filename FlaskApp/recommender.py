#!/bin/python
from movielens import *
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, json, request, jsonify
from scipy.stats.stats import pearsonr

import numpy as np
import pickle
import random
import sys
import time
import json


# Set default encoding since some movie titles are in latin-1 encoding
reload(sys)
sys.setdefaultencoding('utf-8')

user = []
item = []
rating = []

d = Dataset()
d.load_users("data/u.user", user)
d.load_items("data/u.item", item)
d.load_ratings("data/u.base", rating)

n_users = len(user)
n_items = len(item)

# Load smilar movies dictionary
f = open('./data_dumps/similar_movies', 'r')
similar_movies = pickle.load(f)
f.close()

# Utility matrix constructed after performing regularized matrix factorization
utility = np.load("./data_dumps/utility.dat")

# Stores similarity measures of the new user
pcs_arr = np.zeros(943)
	
# Initializing rating array for the new user    
user944 = np.zeros(1682)


# Stores movies rated by the user
rated = {}

user944_copy = np.zeros(1682)

# Create the application object
recommender = Flask(__name__)


# Define the basic route and its corresponding request handler
@recommender.route("/", methods=['POST', 'GET'])
def rate():
	
    # Flag is set until user rates minimum 10 movies
    flag = 1
	
    try:
		k = int(request.args['title'])
		v = int(request.args['rating'])
		rated[k] = v
		user944[k-1] = v
    except Exception:
		pass
	
    if len(rated) >= 10:
		flag = 0
	
    return render_template('RateMovies.html', items = item, flag = flag)


@recommender.route("/genRec")
def genRec():

    unrated = range(1, 1683)
    user944_copy = user944

    # Average of all user ratings
    avg = np.mean(user944)
	
    # Calculate user-user similarity values 
    for i in range(n_users):
		pcs_arr[i] = pearsonr(user944_copy, utility[i])[0]
	
    # Guess ratings of unrated items    
    for i in range(n_items):
		if user944[i] == 0:
			user944_copy[i] = guess(i+1, 150, avg)
	
    # Store ids of only unrated movies        
    unrated = np.setdiff1d(unrated, rated.keys())
	# Store predicated ratings of unrated movies
    unrated_r = np.zeros(len(unrated))
    for i in range(len(unrated_r)):
		unrated_r[i] = user944_copy[unrated[i]-1]

    # Store ids of top unrated movies    
    recommendations = [x for (x,y) in sorted(zip(unrated, unrated_r), key=lambda pair:pair[1], reverse=True)]
	
    return render_template('Recommendation.html', recommend=recommendations[:20], items=item)


@recommender.route("/similarMovies", methods=['POST', 'GET'])
def similarMovies():

    k = int(request.args['title'])

    return render_template('SimilarMovies.html', id=k, movies=similar_movies[k], items=item)



# Calculate similarity between two users
def pcs(x, y, avgx, avgy):
    numerator = 0
    den1 = 	0
    den2 = 0
    for i in range(n_items):
    	if x[i] > 0 and y[i] > 0:
    		a = x[i]
    		b = y[i]
    		numerator += (a - avgx) * (b - avgy) 
    		den1 += (a - avgx) ** 2
    		den2 += (b - avgy) ** 2
    denominator = (den1 ** 0.5) * (den2 ** 0.5)
    if denominator == 0:
    	return 0
    else:
    	return numerator/denominator


# Predict the rating of a movie 
def guess(i_id, top_n, avg):
    
    #finding top n similar users
    similarity = []

    for i in range(0, n_users):
    	similarity.append((pcs_arr[i],i+1))
    similarity.sort(key=lambda x:x[0], reverse=True)
    
    similarity = similarity[:top_n]
    ratings_topn = [(i,utility[i - 1][i_id - 1]) for v,i in similarity if utility[i-1][i_id - 1] > 0]
    ratings_norm = [r - user[u - 1].avg_r for u,r in ratings_topn]
    avg_topn = np.mean(ratings_norm) if len(ratings_norm) > 0 else 0
    r = abs(avg + avg_topn)
    
    if r < 1.0:
    	return 1.0
    elif r > 5.0:
    	return 5.0
    else:
    	return r


# See Movie List
@recommender.route("/movieList")
def movieList():
    return render_template('movieList.html', items=item)

@recommender.route("/ratedList")
def ratedList():
    return render_template('ratedMovies.html', rated=rated, items=item)

if __name__ == "__main__":
	recommender.run()