#Part of Speech Tagger
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def parts_of_speech_tag(tag):
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('N'):
        return wordnet.NOUN
    if tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.VERB

sentence = 'I am a good boy who wants to go there and start my study happily'
print(sentence)

words_and_tags = nltk.pos_tag(sentence.split())
lemmatizer = WordNetLemmatizer()

for word, tag in words_and_tags:
    lemma = lemmatizer.lemmatize(word.lower(),parts_of_speech_tag(tag))
    print(lemma, end=" ")
    