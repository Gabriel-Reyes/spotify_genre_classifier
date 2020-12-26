import pandas as pd
import numpy as np

import setup

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=setup.client_id,
                                                            client_secret=setup.client_secret))


# function searches for artist name, draws first result, and returns random sampling of 100 tracks with attributes

def get_discography(search):
    query = sp.search(q=search, type='artist', limit=1)['artists']['items'][0]

    name = query['name']
    popularity = query['popularity']
    followers = query['followers']['total']
    artist_uri = query['uri']

    album_uris = pd.DataFrame(sp.artist_albums(artist_uri, album_type='album', country='US', limit=50)['items'])['uri']
    
    album_tracks = []

    for album in album_uris:
        tracklist = sp.album_tracks(album, market='US')['items']
        album_tracks +=tracklist

    album_tracks = pd.DataFrame(album_tracks)

    track_attributes = []

    for i in range(1, int(np.ceil(len(album_tracks)/100)) + 1):
        segment = sp.audio_features(album_tracks['uri'][i*100-100:i*100])
        track_attributes += segment

    track_attributes = pd.DataFrame(track_attributes)
    track_attributes['artist'] = name
    track_attributes['artist popularity'] = popularity
    track_attributes['followers'] = followers

    if len(track_attributes) > 100:
        track_attributes = track_attributes.sample(100, random_state=5)

    return track_attributes


# reading in genre database, csv created via genre_database_builder.py file

training_data = pd.read_csv(setup.csv_path)

genres = training_data[['genre', 'genre code']].drop_duplicates().set_index('genre code')['genre'].to_dict()

# genre estimator, prints numerical genre code predictions for each artist, and retuns final df of results

def predictor(artists, model_type):
    
    X_train = training_data.iloc[:, :11]
    y_train = training_data['genre code']
    
    if model_type == 'knn':
        model_name = 'K-Neighbors Classifier'
        model = KNeighborsClassifier()

    if model_type == 'lr':
        model_name = 'Logistic Regression'
        model = LogisticRegression()

    if model_type == 'rfr':
        model_name = 'Random Forest Classifier'
        model = RandomForestClassifier()

    if model_type == 'xgb':
        model_name = 'XG Boost'
        model = XGBClassifier()

    if model_type == 'nn':
        model_name = 'Neural Network'
        model = MLPClassifier()

    model.fit(X_train, y_train)
    
    predictions = {}

    for artist in artists:

        discog = get_discography(artist[0])
        name = discog['artist'].iloc[0]

        X_artist = discog.iloc[:, :11]
        
        prediction = model.predict(X_artist)

        genre_code_guess = np.bincount(prediction).argmax()
        percent = round(np.bincount(prediction)[np.bincount(prediction).argmax()] / len(prediction), 2)

        predictions[name] = {}
        predictions[name]['actual genre'] = genres[artist[1]]
        predictions[name]['predicted genre'] = genres[genre_code_guess]
        predictions[name]['percent'] = percent

    print('Model Type:', model_name)
    print('-'*30)

    return print(pd.DataFrame(predictions).transpose())


# evaluating ML models with one new artist per genre

artists = [('joan jett', 0),
            ('chet baker', 1),
            ('blind willie johnson', 2),
            ('hieroglyphics', 3),
            ('lemon jelly', 4),
            ('nirvana', 0),
            ('lauryn hill', 3),
            ('sonny rollins', 1)]