# movie-rec
Movie recommendation system

**Required/Installation: Python 2.x, numpy, scikit-learn, flask**  

**Dataset: MovieLens 100k dataset**  
The dataset and it's details can be found here - https://grouplens.org/datasets/movielens/100k/  

**Create folder called "data" inside the FlaskApp and ClusteringMethod folders and Downlaod the dataset and extract it's contents there.**  

Movie recommendation system built with Python 2.x and Flask using collaborative filtering and matrix factorization   

### FlaskApp
* Directly Run recommender.py and test the app on localhost
* To create data_dumps again and check MSE
     1. Run SVD_Modified.py
     2. Run recommender.py
     
### ClusteringMethods
Modelling a recommendation system using k-means clustering based on the learnings from Udacity's ud741 course  
* Run ItemClustering.py for clustering movies based on genre
* Run UserClustering.py for clustering users based on ratings
