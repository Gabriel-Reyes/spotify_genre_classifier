from main import get_discography
import setup

import pandas as pd

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=setup.client_id,
                                                            client_secret=setup.client_secret))

## creating database of different genres, for fitting ML models

# classic rock

beatles = get_discography('the beatles')
led_zep = get_discography('led zep')
stones = get_discography('rolling stones')
floyd = get_discography('pink floyd')
ccr = get_discography('creedence clearwater revival')
queen = get_discography('queen')
acdc = get_discography('AC DC')
who = get_discography('the who')
sabbath = get_discography('black sabbath')
hendrix = get_discography('jimi hendrix')

# jazz

miles = get_discography('miles davis')
coltrane = get_discography('john coltrane')
monk = get_discography('thelonious monk')
ellington = get_discography('duke ellington')
evans = get_discography('bill evans')
brubeck = get_discography('dave brubeck')
montgomery = get_discography('wes montgomery')
mingus = get_discography('charles mingus')
hancock = get_discography('herbie hancock')
holiday = get_discography('billie holiday')

# blues

bbking = get_discography('bb king')
howling = get_discography('howling wolf')
buddy = get_discography('buddy guy')
johnson = get_discography('robert johnson')
srv = get_discography('stevie ray vaughan')
jlhooker = get_discography('john lee hooker')
albert = get_discography('albert king')
tbone = get_discography('t bone walker')
otisrush = get_discography('otis rush')
freddy = get_discography('freddy king')

# hiphop

delasoul = get_discography('de la soul')
outkast = get_discography('outkast')
tribe = get_discography('a tribe called quest')
mosdef = get_discography('mos def')
mobbdeep = get_discography('mobb deep')
talib = get_discography('talib kweli')
krsone = get_discography('krs one')
pharcyde = get_discography('the pharcyde')
mfdoom = get_discography('mf doom')
jurassic = get_discography('jurassic 5')

# downtempo

thievery = get_discography('thievery corporation')
nightmares = get_discography('nightmares on wax')
massive = get_discography('massive attack')
kruder = get_discography('kruder and dorfmeister')
zero7 = get_discography('zero 7')
portishead = get_discography('portishead')
morcheeba = get_discography('morcheeba')
royksopp = get_discography('royksopp')
telepopmusik = get_discography('telepopmusik')
boozoo = get_discography('boozoo bajou')


# grouping all artists samplings into unique genres, selecting common genre tags and respective numerical code

rock = ['rock', 0, [beatles, led_zep, stones, floyd, ccr, queen, acdc, who, sabbath, hendrix]]
jazz = ['jazz', 1, [miles, coltrane, monk, ellington, evans, brubeck, montgomery, mingus, hancock, holiday]]
blues = ['blues', 2, [bbking, howling, buddy, johnson, srv, jlhooker, albert, tbone, otisrush, freddy]]
hiphop = ['hip hop', 3, [delasoul, outkast, tribe, mosdef, mobbdeep, talib, krsone, pharcyde, mfdoom, jurassic]]
downtempo = ['downtempo', 4, [thievery,  nightmares, massive, kruder, zero7, portishead, morcheeba, royksopp, telepopmusik, boozoo]]


genres = [rock, jazz, blues, hiphop, downtempo]


# compiling all genres into single df, with genre and resptive encoding

def genre_compiler(genres):
    df = []

    for genre in genres:
    
        for artist in genre[2]:
            artist['genre'] = genre[0]
            artist['genre code'] = genre[1]
            df.append(artist)

    return pd.concat(df)

training_data = genre_compiler(genres)

training_data.to_csv('training_data.csv', index=False)