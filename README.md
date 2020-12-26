# spotify_genre_classifier

This program is a simple prototype to evaluate the effectiveness of Spotify track audio features to classify artist genres.

The genre database builder file retreives the discography from all selected artists, sampling 100 songs at random from each, then combining and exporting into a single csv file with manually selected genres and genre encodings.

The main file contains the genre predictor which allows for a list of new artists, genre codes, and classifer to be selected. Models are fitted with the training data csv generated from the genre database builder.

The example file is a jupyter notebook that shows results for a small group of "unseen" artists to show functionality and accuracy between classification models.

Without any tuning, decision trees seem to be the most effective at correctly identifying artist genres.