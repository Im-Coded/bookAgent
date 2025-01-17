from transformers import pipeline
import schedule
import time
from ..twitter_bot.twitter_client import TwitterBot
from ..airdrop.bindings import AirdropManager
from .neural_network import MarketAnalysisNetwork
from .market_data import MarketDataFetcher
import torch
import numpy as np
from transformers import GPT2LMHeadModel

class AIAgent:
    def __init__(self):
        # Initialize components
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.twitter_bot = TwitterBot()
        self.airdrop_manager = AirdropManager()
        self.market_data_fetcher = MarketDataFetcher()
        
        # Neural models initialization
        self.market_network = MarketAnalysisNetwork()
        self.gpt_model = GPT2LMHeadModel.from_pretrained\('gpt2')
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Learning parameters
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.market_network.parameters(), 
                                        lr=self.learning_rate)
        
    def _fetch_market_data(self) -> np.ndarray:
        return self.market_data_fetcher.fetch_data()
        
    def _create_prompt(self, sentiment: float) -> str:
        market_condition = "bullish" if sentiment > 0.5 else "bearish"
        return f"Create an engaging crypto market update tweet. Market sentiment is {market_condition}. Current analysis shows"
        
    def analyze_market_sentiment(self) -> float:
        market_data = self._fetch_market_data()
        market_tensor = torch.FloatTensor(market_data).unsqueeze(0)
        
        with torch.no_grad():
            sentiment_score = self.market_network(market_tensor)
        return sentiment_score.item()
        
    def generate_tweet(self) -> str:
        sentiment = self.analyze_market_sentiment()
        prompt = self._create_prompt(sentiment)
        
        inputs = self.gpt_tokenizer(prompt, return_tensors='pt')
        outputs = self.gpt_model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9
        )
        
        return self.gpt_tokenizer.decode(outputs[0])
        
    def schedule_tasks(self):
        schedule.every().day.at("10:00").do(self.daily_tweet)
        schedule.every().week.do(self.weekly_airdrop)
        
    def daily_tweet(self):
        tweet = self.generate_tweet()
        success = self.twitter_bot.post_tweet(tweet)
        if success:
            print(f"Successfully posted tweet: {tweet}")
        
    async def weekly_airdrop(self):
        holders = await self.airdrop_manager.get_eligible_holders()
        amount = self._calculate_airdrop_amount()
        await self.airdrop_manager.send_airdrop(holders, amount)
        
    def _calculate_airdrop_amount(self) -> int:
        # Implement airdrop amount calculation based on treasury
        return 100 * 10**18  # 100 tokens in wei 
