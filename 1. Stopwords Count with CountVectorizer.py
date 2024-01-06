#Find out the number of stopwords from several sentences
from sklearn.feature_extraction.text import CountVectorizer
import string

#Sentences
sentences = ['The quick brown fox jumps over the lazy dog',
             'She is as smart as she is diligent in her work.',
             'He has a plan that he thinks is good for all',
             'It is what it is, and that is the truth.',
             'They were going to the market to buy some fruits.']

#Remove Punctuation
for punc in string.punctuation:
    for i in range(len(sentences)):
        sentences[i] = sentences[i].replace(punc, ' ')

# Initialize CountVectorizer with English stop words
Vectorizer = CountVectorizer(stop_words='english')

#Fit and Transform the text
Vectorizer.fit_transform(sentences)

# Found all the words without stopwords 
word_list_without_stopword = Vectorizer.get_feature_names_out()

# Final Result
for i in range(len(sentences)):
    print('For Sentence No.', i, '::')
    check = []
    for w in sentences[i].split():
        if w not in word_list_without_stopword and w.lower() not in check:
            print(w.lower())
            check.append(w.lower())

