# Movie Recommendation System
import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def from_dictionary(sentence):
    result = ''
    for item in ast.literal_eval(sentence):
        result = result + ' ' + item['name']
    return result

dataset = pd.read_csv('tmdb_5000_movies.csv')
list_dataset = []

for i in range(len(dataset)):
    sentence = ' '
    sentence = sentence + ' ' + from_dictionary(dataset['genres'][i])
    sentence = sentence + ' ' + from_dictionary(dataset['keywords'][i])
    sentence = sentence + ' ' + from_dictionary(dataset['production_countries'][i])
    sentence = sentence + ' ' + str(dataset['overview'][i])
    sentence = sentence + ' ' + str(dataset['popularity'][i])
    sentence = sentence + ' ' + str(dataset['tagline'][i])
    sentence = sentence + ' ' + str(dataset['vote_average'][i])
    list_dataset.append(sentence)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(list_dataset)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

recommended_movie = 'Scream 3'

recommended_movie_index = -1
for i in range (len(dataset)):
    if recommended_movie == dataset['title'][i]:
        recommended_movie_index = i
        break

recommended_movie_list_sim = cosine_sim[recommended_movie_index]
result_list = sorted(enumerate(recommended_movie_list_sim), key=lambda x: x[1])
result_index_list = [index for index, value in result_list]

print('Recommended Top Five Movies for',recommended_movie,':: ')
for i in range(2,7):
    print(dataset['title'][result_index_list[len(result_index_list)-i]])