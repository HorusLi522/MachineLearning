from bs4 import BeautifulSoup

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import re

class CleanText:
	'''
	Attempts at cleaning text were made with modules lxml 
	(clean_html),Goose3, Newspaper3k and the preferred 
	method I've used below. There is still some junk that 
	gets added but better than the above methods.

	Params:
		raw_html (str) = raw html doc, e.g. 
						 requests.get(URL).text
	
	Returns:
		All words on the page, with options to stem
		lemmatize and remove stop words

	Example:
		url = 'http://www.bbc.co.uk'		 
		html = requests.get(url)
		ct = CleanText(html.content)		
		>>>ct.tokenize(stemmer='Snowball',lemmatize=True)
		   bbc home homepag access link skip content access 
		   help bbc id notif home news sport weather iplay 
		   tv radio cbbc cbeebi food bites music earth art 
		   make digit taster local menu search search bbc
		   ...
	'''
	def __init__(self,raw_html):
		self.raw_html = raw_html
		self.soup = BeautifulSoup(self.raw_html,'html.parser')
		self.stopset = set(stopwords.words('english'))
		self.no_scripts = self._script_removed()
		self.stemmers = {'Snowball':SnowballStemmer('english'),
						 'Porter':PorterStemmer()}

	def _script_removed(self):
		'''
		Credit: https://stackoverflow.com/questions/\
				328356/extracting-text-from-html-file\
				-using-python
		Returns:
			All text from self.soup __init__ variable 		
		'''
		
		for script in self.soup(["script", "style"]):
			script.extract()    
		
		text = self.soup.get_text(separator=u' ')

		lines = (line.strip() for line in text.splitlines())

		chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

		text = ' '.join(chunk for chunk in chunks if chunk)

		return text

	def tokenize(self,stop_words=True,stemmer=False,lemmatize=False):
		'''
		Tokenizes text (bit of a misnomer because it returns a str)
		
		Params:
			stop_words (bool): Enables stop_words to occur 
							   to self.no_scripts at __init__
			stemmer (bool): Enables stemmer to occur 
							   to self.no_scripts at __init__
							   Options include 'Snowball' or
							   'Porter'
			lemmatize (bool): Enables lemmatize to occur 
							   to self.no_scripts at __init__
		Returns:
			A string of all text from the webpage
		'''

		tokens = word_tokenize(str(self.no_scripts))		
		tokens = [re.compile('\w+').findall(w) for w in tokens]	
		tokens = [x[0].lower() for x in tokens if x]
		
		if stop_words:
			tokens = [w for w in tokens if not w in self.stopset]		
		if lemmatize:
			wordnet_lemmatizer = WordNetLemmatizer()
			tokens = [wordnet_lemmatizer.lemmatize(w) for w in tokens]
		if stemmer:
			stem_obj = self.stemmers[stemmer]
			tokens = [stem_obj.stem(w) for w in tokens]
			
		return " ".join(tokens)

