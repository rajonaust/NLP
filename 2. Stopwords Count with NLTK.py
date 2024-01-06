#Find out the number of stopwords from several sentences
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

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

all_stopwords = stopwords.words('english')

# Final Result
for i in range(len(sentences)):
    print('For Sentence No.', i, '::')
    check = []
    for w in sentences[i].split():
        if w.lower() in all_stopwords and w.lower() not in check:
            print(w.lower())
            check.append(w.lower())
