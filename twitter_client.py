from tweepy import Client, OAuth1UserHandler, Tweet
import os
from dotenv import load_dotenv
from typing import Dict, Optional

class TwitterBot:
    def __init__(self):
        load_dotenv()
        self.client = Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
            wait_on_rate_limit=True
        )

    def post_tweet(self, message: str) -> bool:
        try:
            response = self.client.create_tweet(text=message)
            print(f"Tweet posted successfully with iD: {response.data['id']}")
            return True
        except Exception as e:
            print(f"Error while posting tweet: {e}")
            return False

    def analyze_engagement(self, tweet_id: str) -> Dict:
        try:
            tweet = self.client.get_tweet(
                tweet_id,
                tweet_fields=['public_metrics']
            )
            
            if tweet.data:
                metrics = tweet.data.public_metrics
                return {
                    'retweets': metrics['retweet_count'],
                    'replies': metrics['reply_count'],
                    'likes': metrics['like_count'],
                    'quotes': metrics['quote_count']
                }
            return {}
        except Exception as e:
            print(f"Error analyzing engagement: {e}")
            return {}
            
    def get_trending_topics(self, woeid: int = 1) -> list:
        try:
            trends = self.client.get_place_trends(woeid)
            return [trend['name'] for trend in trends[0]['trends']]
        except Exception as e:
            print(f"Error fetching trends: {e}")
            return [] 
