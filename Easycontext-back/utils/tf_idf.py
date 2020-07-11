

import nltk #Natural Language processing toolkit
import os
import pickle
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
stopwords=stopwords.words('English') #To eliminate stopwords 
import string
import re
import math






def preprocess(documents):

    document_to_senctence_corpus = {}
    #punctuation = list(string.punctuation)
    #ps=PorterStemmer()
    WNlemma = nltk.WordNetLemmatizer()
    i=0
    modals = ['can', 'could', 'may', 'might', 'must', 'will'] #To remove the modals
    for each_doc in documents:
        x=[]
        each_doc = each_doc.lower()
        l = sent_tokenize(each_doc)
        l2= word_tokenize(each_doc)
        for w in l2:
            if w not in stopwords:
                
                x.append(WNlemma.lemmatize(w))
            
        
        fileText = " ".join(x)
        
        document_to_senctence_corpus[i]=(fileText)
        i+=1
    os.chdir("..")
    return document_to_senctence_corpus



import operator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
#nltk.download('averaged_perceptron_tagger')
Stopwords = set(stopwords.words('english'))
wordlemmatizer = WordNetLemmatizer()
def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(wordlemmatizer.lemmatize(word))
    return lemmatized_words
def stem_words(words):
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text
def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
        if word not in words_unique:
            words_unique.append(word)
    for word in words_unique:
        dict_freq[word] = words.count(word)
    return dict_freq
def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word,tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
             pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb





from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
def tfidf_topicextraction(x):
    """
    This function returns idf -Inverse document frequency
    """
    text = preprocess(x)
    vect = TfidfVectorizer(ngram_range=(1,1)).fit([text])
    
    X_train_vectorized = vect.transform([text])
    
    feature_names = np.array(vect.get_feature_names())
    
    idfs = vect.idf_
    sorted_tfidf_index = idfs.argsort()
     
    smallest_dict = {}
    for index in sorted_tfidf_index[:-30:-1]:
        smallest_dict[feature_names[index]] = idfs[index]
    
    largest_dict = {}
    for index in sorted_tfidf_index[:30:1]:
        largest_dict[feature_names[index]] = idfs[index]
    
    smallest_series = pd.Series(smallest_dict)
    largest_series = pd.Series(largest_dict)
    
    
    return (smallest_series.sort_values(ascending=True), largest_series.sort_values(ascending=False))


#Tf-idf _ summarizer_utils

def tf_score(word,sentence):
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf
def idf_score(no_of_sentences,word,sentences):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        sentence = remove_special_characters(str(sentence))
         
        sentence = sentence.split()
        sentence = [word for word in sentence if word.lower() not in Stopwords and len(word)>1]
        sentence = [word.lower() for word in sentence]
        sentence = [wordlemmatizer.lemmatize(word) for word in sentence]
        if word in sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    idf = math.log10(no_of_sentences/no_of_sentence_containing_word)
    return idf
def tf_idf_score(tf,idf):
    return tf*idf
def word_tfidf(dict_freq,word,sentences,sentence):
    word_tfidf = []
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf
def sentence_importance(sentence,dict_freq,sentences):
    sentence_score = 0
    sentence = remove_special_characters(str(sentence)) 
    sentence = re.sub(r'\d+', '', sentence)
    pos_tagged_sentence = [] 
    no_of_sentences = len(sentences)
    pos_tagged_sentence = pos_tagging(sentence)
    for word in pos_tagged_sentence:
        if word.lower() not in Stopwords and word not in Stopwords and len(word)>1: 
            word = word.lower()
            word = wordlemmatizer.lemmatize(word)
            sentence_score = sentence_score + word_tfidf(dict_freq,word,sentences,sentence)
    return sentence_score


# ## Summary using TF-IDF

# In[13]:

from gensim.parsing.preprocessing import STOPWORDS

def main_tfidf(text):
    #file1=file
    
    #file = os.getcwd()+ "/" + file1
    #file = open(file , 'r')
    #text = file.read()
    tokenized_sentence = sent_tokenize(text)
    text = remove_special_characters(str(text))
    text = re.sub(r'\d+', '', text)
    tokenized_words_with_stopwords = word_tokenize(text)
    tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(tokenized_words)
    word_freq = freq(tokenized_words)
    input_user = 40
    no_of_sentences = int((input_user * len(tokenized_sentence))/100)
    #print(no_of_sentences)
    c = 1
    sentence_with_importance = {}
    for sent in tokenized_sentence:
        sentenceimp = sentence_importance(sent,word_freq,tokenized_sentence)
        sentence_with_importance[c] = sentenceimp
        c = c+1
    sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1),reverse=True)
    cnt = 0
    summary = []
    sentence_no = []
    for word_prob in sentence_with_importance:
        if cnt < no_of_sentences:
            sentence_no.append(word_prob[0])
            cnt = cnt+1
        else:
            break
    sentence_no.sort()
    cnt = 1
    for sentence in tokenized_sentence:
        if cnt in sentence_no:
            summary.append(sentence)
        cnt = cnt+1
    summary = " ".join(summary)
    print("\n")
    print("Summary:")
    print(summary)
    #summary_file="summary_"+(file1)
    #outF = open(summary_file,"w")
    #outF.write(summary)
    return summary












