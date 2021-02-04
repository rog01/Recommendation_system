#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 08:42:56 2020

@author: roger"""

from flask import Flask

from flask import render_template, request
from lightfm import LightFM
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from sklearn.metrics.pairwise import linear_kernel
from copy import deepcopy
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pickle

app = Flask(__name__)

def preprocessing():
    global ratings, artists, ratings_df, ap, artist_rank
    plays = pd.read_csv('data/user_artists.dat', sep='\t')
    artists = pd.read_csv('data/artists.dat', sep='\t', usecols=['id','name'])
    
    # Merge artist and user pref data
    ap = pd.merge(artists, plays, how="inner", left_on="id", right_on="artistID")
    ap = ap.rename(columns={"weight": "playCount"})
    
    # Group artist by name
    artist_rank = ap.groupby(['name']) \
        .agg({'userID' : 'count', 'playCount' : 'sum'}) \
        .rename(columns={"userID" : 'totalUsers', "playCount" : "totalPlays"}) \
        .sort_values(['totalPlays'], ascending=False)
    
    artist_rank['avgPlays'] = artist_rank['totalPlays'] / artist_rank['totalUsers']
    # Preprocessing
    pc = ap.playCount
    play_count_scaled = (pc - pc.min()) / (pc.max() - pc.min())
    ap = ap.assign(playCountScaled=play_count_scaled)
    
    # Build a user-artist rating matrix 
    ratings_df = ap.pivot(index='userID', columns='artistID', values='playCountScaled')
    ratings = ratings_df.fillna(0).values 

# with open('preprocessing.pkl','rb') as file:
#       ratings, artists, ratings_df, ap, artist_rank = pickle.load(file)

def get_recommendation_LightFM(artist):
    global ratings, artists, ratings_df, ap, artist_rank

    preprocessing()
    
    #pickle.dump(preprocessing(),open('preprocessing.pkl','wb'))
    
    if artist != '':  
        new_user_rating  = np.zeros(ratings.shape[1])
        for art in artist:
            art = art.replace("_", " ")
            #rajout d'un nouvel utilisateur dans la base avec la cotation a 1 de son artist prefere dans ratings
            
            if artists[artists['name']==art].empty == False:
                artist_col = artists[artists['name']==art].index[0] 
            else :
                artist_col = 300 #Black Eyed Peas
            new_user_rating[artist_col]=1
            
        ratings = np.append(ratings, [new_user_rating], axis=0) 

    # Build a sparse matrix
    X = csr_matrix(ratings)
    
    n_users, n_items = ratings_df.shape        
   
    artist_names = ap.sort_values("artistID")["name"].unique()
    
    # Build data references + train test
    Xcoo = X.tocoo() #coordonnees xsparse de ratings
    data = Dataset() #lighhtFM dataset
    data.fit(np.arange(n_users+1), np.arange(n_items))
    interactions, weights = data.build_interactions(zip(Xcoo.row, Xcoo.col, Xcoo.data)) 
    train, test = random_train_test_split(interactions)
    
        # Train
    model = LightFM(no_components = 80,
                        k = 20,
                        n = 80,
                        learning_schedule = 'adadelta',
                        learning_rate = 0.005,
                        loss='warp') 
    model.fit(train, epochs=30, num_threads=2)
        # Predict
    if artist == '':
        scores = model.predict(0, np.arange(n_items))
    else :
        scores = model.predict(n_users, np.arange(n_items))
    top_items = artist_names[np.argsort(-scores)]
    return top_items

def get_top_artists_listened():
    artist = ''
    get_recommendation_LightFM(artist)
    artist_names_listened = artist_rank.sort_values("totalUsers", ascending = False).index.unique()[0:10]
    return artist_names_listened

# def get_top_artists_recommended():
#     artist = ''
#     top_items, ap, artist_rank, ratings = get_recommendation_LightFM(artist)
#     artist_names_rec = ap.sort_values("playCount", ascending = False)["name"].unique()
#     return artist_names_rec

# Function that takes in user_id as input and outputs most similar users and the artits ratings
def get_recommendations(user_id, cosine_sim):    
    global ratings, ap
    # Get the pairwsie similarity scores of all users with that user
    sim_scores = list(enumerate(cosine_sim))
    # Sort the users based on the similarity ratings of the artists
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 50 most similar users
    sim_scores = sim_scores[1:11]
    # Get the user indices similar to the user in parameter
    user_indices = [i[0] for i in sim_scores] 
    artist_top = []
    idx_i_list=[]
    ratings_ = deepcopy(ratings)
    for user in user_indices:
        # take the 5 most important artists for this user
        for j in range(5):
            idx_i = np.argmax(ratings_[user])
            idx_i_list.append(idx_i)
            ratings_[user][idx_i]=-1  
        for i, idx in enumerate(idx_i_list):
            if len(artists[artists['id']==idx].index) == 1:
                if ap['name'][ap['artistID']==idx].unique()[0] not in artist_top:
                    artist_top.append(ap['name'][ap['artistID']==idx].unique()[0]) 
            
    # Return the top 10 most similar users ratings of artists
    return artist_top

def cosine_similarity(user_id):
    global ratings    
    X_matrix_svd = get_X_matrix_svd(ratings)
    cos_sim = linear_kernel(X_matrix_svd,X_matrix_svd[user_id].reshape(1, -1))
    return cos_sim

def get_X_matrix_svd(ratings):
    # Build a sparse matrix
    X = csr_matrix(ratings)
    svd = TruncatedSVD(n_components=100, n_iter=3)
    X_matrix_svd = svd.fit_transform(X)
    return X_matrix_svd    

def get_top_artists_recommended_svd(user_id):
    preprocessing()     
    artist_names_rec = get_recommendations(user_id, cosine_similarity(user_id))
    return artist_names_rec

def most_rec_artists():
    global n_users
    artist_names_rec_hist = []
    for i in range(100):
        userID = np.random.randint(n_users)
        artist_names_rec_hist.append(get_top_artists_recommended_svd(userID))
        
    artist_names_rec_hist = [item for sublist in artist_names_rec_hist for item in sublist]
    return artist_names_rec_hist

def artist_dist(artists):
    distribution  = {}
    for  elt in artists:     
        distribution[elt] = round(artists.count(elt)/len(artists),5)*100
    return distribution

# def graph(artists_dist_sorted):  
#     name = []
#     data = []
#     for i  in range(len(artists_dist_sorted)):
#         name.append(artists_dist_sorted[i][0])
#         data.append(artists_dist_sorted[i][1])

#     explode=(0, 0.15, 0, 0,0,0,0,0,0)
#     plt.pie(data, explode=explode, labels=name, autopct='%1.1f%%', startangle=90, shadow=True)
#     plt.axis('equal')
    
with open('most_rec_artists.pkl','rb') as file:
    artist_names_rec_hist = pickle.load(file)
    
with open('artist_names_listened.pkl','rb') as file:
    artist_names_listened = pickle.load(file)
    
@app.route('/')
def index():
    artist = ''
    top_items = get_recommendation_LightFM(artist)        
    return render_template('layout.html', result=top_items[0:15])

@app.route('/search/', methods = ['GET', 'POST'])
def search():
    if request.method == 'POST':
        artist = [request.form['artist'].capitalize()]
        top_items = get_recommendation_LightFM(artist)
    return render_template('layout.html', result=top_items[0:15])

@app.route('/search_multiple/', methods = ['GET', 'POST'])
def search_multiple():
    if request.method == 'POST':
        artist = request.form.getlist('result')
        #print(artist)
        
        top_items = get_recommendation_LightFM(artist)
    return render_template('layout.html', result=top_items[0:15])

@app.route('/top_artist/')
def top_artist():
    #artist_names_listened = get_top_artists_listened()
    #pickle.dump(get_top_artists_listened(),open('artist_names_listened.pkl','wb'))
    
    #artist_names_rec = get_top_artists_recommended_svd(0)
    #artist_names_rec_hist = most_rec_artists()
    #pickle.dump(most_rec_artists(),open('most_rec_artists.pkl','wb'))
    
    artists_dist = artist_dist(artist_names_rec_hist)
    artists_dist_sorted = sorted(artists_dist.items(), key=lambda x: x[1], reverse=True)
    #print(artists_dist_sorted)
    return render_template('layout_stat.html', artist_names_listened=artist_names_listened[0:9],
                           artist_names_rec=artists_dist_sorted[0:9])
if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)