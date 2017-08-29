import tweepy
from textblob import TextBlob

consumer_key = 'CONSUMER_KEY_HERE'
consumer_secret = 'CONSUMER_SECRET_HERE'

access_token = 'ACCESS_TOKEN_HERE'
access_token_secret = 'ACCESS_TOKEN_SECRET_HERE'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

for tweet in public_tweets:
    print(tweet.text)

    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    print("")