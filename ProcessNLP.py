import pandas as pd
import numpy as np
import requests
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import re
import nltk

nltk.download('punkt')
nltk.download("stopwords")

dataset = pd.read_excel('data.xlsx')
#dataset.head()

dataset = dataset[[ 'URL']]

URLs = dataset['URL'].tolist()

articles = []
for url in URLs:
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")
    articles.append(soup.get_text())
#print('done')

stop_words = set(stopwords.words('english'))
master_dictionary = pd.read_excel('master_dictionary.xlsx')

positive_dictionary = [x for x in master_dictionary[master_dictionary['Positive'] != 0]['Word']]
negative_dictionary = [x for x in master_dictionary[master_dictionary['Negative'] != 0]['Word']]

# Functions
def tokenize_it(text):
    text = re.sub(r'[^A-Za-z]',' ',text.upper())
    tokenized_words = word_tokenize(text)
    return tokenized_words

def noStopwords(words, stop_words):
    return [x for x in words if x not in stop_words]

def positive_score(words):
    score = 0
    for word in words:
        if word in positive_dictionary:
            score+=1
    return score

def negative_score(words):
    score = 0
    for word in words:
        if word in negative_dictionary:
            score+=1
    return score

def polarity(Positive_score, negative_score):
    return (Positive_score - negative_score)/((Positive_score + negative_score)+ 0.000001)

def subjectivity(Positive_score, negative_score, num_words):
    return (Positive_score+negative_score)/(num_words+ 0.000001)

def is_Complex(word):
    if(len(word) > 2 and (word[-2:] == 'es' or word[-2:] == 'ed')):
        return False
    
    count =0
    vowels = ['a','e','i','o','u']
    for i in word:
        if(i.lower() in vowels):
            count = count +1
        
    if(count > 2):
        return True
    else:
        return False

def fogIndex(average_sentence_length, percentage_complexwords):
    return 0.4*(average_sentence_length + percentage_complexwords)

def average_words_per_sentence(words):
    list_of_Sentence = sent_tokenize(words)
    sentenceLength = len(list_of_Sentence)
    wordLength = 0
    for i in range(sentenceLength):
        y = word_tokenize(list_of_Sentence[i])
        wordLength += len(y)
    return wordLength//sentenceLength

def pronoun_count(data):
    count = 0
    x = ['I', 'i', 'We', 'we', 'Me', 'me','Ours', 'ours', 'us']
    words = word_tokenize(data)
    for word in range(len(words)):
        if word in x:
            count+=1
    return count

# Dataframe creation
features = ['Positive_score',
      'negative_score',
      'polarity_score',
      'subjectivity_score',
      'average_sentence_length',
      'percentage_of_complex_words',
      'fog_index',
      'complex_word_count',
      'word_count',
      'average_word_length',
      'avg_words_per_sentence',
      'pronouns_count']

for column in features:
    dataset[column] = np.nan


for data in range(len(articles)):
    tokenized_words = tokenize_it(articles[data])               
    words = noStopwords(tokenized_words, stop_words)
    num_words = len(words)

    Positive_score = positive_score(words)
    Negative_score = negative_score(words)                
    polarity_score = polarity(Positive_score, Negative_score)            
    subjectivity_score = subjectivity(Positive_score, Negative_score, num_words)
    avg_words_per_sentence = average_words_per_sentence(articles[data])
    pronouns_count = pronoun_count(articles[data]) 

    sentences = sent_tokenize(articles[data])
    numofSentences = len(sentences)
    average_sentence_length = num_words/numofSentences  

    word = word_tokenize(articles[data])
    count = 0
    for x in word:
        count+=len(x)
    average_word_length = count/num_words

    numofComplexword = 0                         
    for word in words:
        if(is_Complex(word)):
            numofComplexword = numofComplexword+1                       
               
    percentage_complexwords = numofComplexword/num_words 

    fog_index = fogIndex(average_sentence_length, percentage_complexwords)
    
    dataset['Positive_score'].values[data] = Positive_score
    dataset['negative_score'].values[data] = Negative_score
    dataset['polarity_score'].values[data] = polarity_score
    dataset['subjectivity_score'].values[data] = subjectivity_score
    dataset['average_sentence_length'].values[data] = average_sentence_length
    dataset['fog_index'].values[data] = fog_index
    dataset['complex_word_count'].values[data] = numofComplexword
    dataset['percentage_of_complex_words'].values[data] = percentage_complexwords
    dataset['word_count'].values[data] = num_words
    dataset['average_word_length'].values[data] = average_word_length
    dataset['avg_words_per_sentence'].values[data] = avg_words_per_sentence
    dataset['pronouns_count'].values[data] = pronouns_count  
    
dataset.to_excel('assignment_output.xlsx')