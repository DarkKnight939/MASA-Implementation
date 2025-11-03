from utils.data_downloader import download_market_data
from config import Config 

config = Config()
config.market_name = 'DJIA'
config.topK = 10
config.freq = '1d'
config.dataDir = './data'

download_market_data(config)
