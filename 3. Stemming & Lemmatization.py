# Stemming & lemmatization

import nltk
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')

sentence = 'The children joyful laughter, echoing across the park, was heartwarming.'
for punc in string.punctuation:
    sentence = sentence.replace(punc, ' ')

stemming = PorterStemmer()
lemmatizer = WordNetLemmatizer()

for wm in sentence.split():
    w = wm.lower()
    print('Main Word:',w,'Stemming:',stemming.stem(w),'Lemmatization:',lemmatizer.lemmatize(w,pos=wordnet.VERB))
