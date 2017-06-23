import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

class MySpider(scrapy.Spider):
	'''
	Forms the class that the Spider class below
	will accept to run the spider		
	'''	
	name = None
	start_urls = None

class Spider:
	'''
	Crawls urls using Scrapy

	Args:
		name (str): name of spider.
		parse (func): parse object that the spider
					  will use to manipulate scraped
					  data.
	  	start_urls (list): list of urls to scrape. If
	  				       next is used in the parse 
	  				       function the urls will be 
	  				       used to seed the crawl.
		kwargs (dict): list of arguments for spider

	Returns:
		run (func): Scrapy spider

	Example:
		def parse(self,response):
    		yield {
    				'content': response,
            		'url': response.url,
            		'title': response.css('title').extract(),
               	   }

   	   	START_URLS = ['https://www.seerinteractive.com',
   	   				  'https://www.cnn.com']
		
		FILE_NAME = 'my_data.json'

		SETTINGS = {
				    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
				    'FEED_FORMAT': 'json',
					'FEED_URI': FILE_NAME,
					'JOBDIR': './job', #saves job history to avoid duplicating efforts	
					} 
		s = Spider(parse=parse,		  
				   start_urls=START_URLS,
		  		   settings=SETTINGS)
		s.run()  

		>>> 2017-06-22 18:02:10 [scrapy.utils.log] INFO: Scrapy 1.4.0 started (bot: scrapybot)
		>>> 2017-06-22 18:02:10 [scrapy.utils.log] INFO: Overridden settings: {'FEED_FORMAT': 'json', 'FEED_URI': 'best_really_cool.json', 'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'}
		>>> 2017-06-22 18:02:10 [scrapy.middleware] INFO: Enabled extensions:
	   	>>> ...
		>>> 'scheduler/enqueued': 1,
		>>> 'scheduler/enqueued/disk': 1,
		>>> 'start_time': datetime.datetime(2017, 6, 22, 22, 2, 10, 239172)}
		>>> 2017-06-22 18:02:11 [scrapy.core.engine] INFO: Spider closed (finished)			  
	'''

	def __init__(self,parse,start_urls,**kwargs):
		self.name = 'spider'
		self.parse = parse		
		self.start_urls = start_urls
		self.kwargs = kwargs.get('settings')

	def run(self):

		process = CrawlerProcess(self.kwargs)
		MySpider.name = self.name
		MySpider.start_urls = self.start_urls
		MySpider.parse = self.parse		
		process.crawl(MySpider)
		process.start() 
