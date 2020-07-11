import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()


import sys
import datetime

import gensim #For Text processing
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords #NlTK's Stopwords
import numpy as np #Numpoy for array's manipulation
import pandas as pd #Pandas for Data Visulization
Stopwords=stopwords.words('English') #To eliminate stopwords 
import string
import unicodedata
import re
import math
#NLTK TOKENIZER
from nltk.tokenize import sent_tokenize,word_tokenize
import operator
from nltk.stem import WordNetLemmatizer #NLTK LEMMATIZER
from nltk.stem.porter import PorterStemmer #NLTK STEMMER
#Other utilities
from nltk.tokenize import sent_tokenize,word_tokenize
Stopwords=(stopwords.words('english'))
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Phrases
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
import operator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
#nltk.download('averaged_perceptron_tagger')
Stopwords = set(stopwords.words('english'))
wordlemmatizer = WordNetLemmatizer()
stemmer=PorterStemmer()
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from summa import keywords

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
    res=[]
    pos_tagged_noun = []
    for word,tag in pos_tag:
        #print()
        #print(word,tag)
        #print()
        if tag == "NN" :
             pos_tagged_noun.append(word)
    res.append(TreebankWordDetokenizer().detokenize(pos_tagged_noun))
    return res

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return (new_words)
stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    ps = PorterStemmer() 
    lem = WordNetLemmatizer().lemmatize(text)
    stem = ps.stem(lem)
    return lem

def preprocess_to_str(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    f_res= ' '.join(result)
    return f_res
def preprocess_to_str_title(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
        
            result.append(lemmatize_stemming(token))
    f_res= ' '.join(result)
    return f_res
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(' '.join(pos_tagging(text))):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    f_res= ' '.join(result)
    return result

def print_topics(model, count_vectorizer, n_top_words):
    topics=[]
  
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        #print("\nTopic #%d:" % topic_idx)
        topics.append((" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])))
    return topics

def lda_t(text):
    Tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    tfidf_data = Tfidf_vectorizer.fit_transform([text])
    number_topics = 1
    number_words = 10

    lda_tfidf = LDA(n_components=number_topics , doc_topic_prior=1, topic_word_prior=1, max_iter=2 , )

    lda_tfidf.fit(tfidf_data)

    y=print_topics(lda_tfidf, Tfidf_vectorizer, number_words)
    return y
def process(Text):
    processed_text=[]
    
    for i in range (len(Text.split('.'))):
        
        sent=Text.split('.')[i].lower()
        sent=re.sub('[,\!?]', '.', sent)
        sent=' '.join(remove_non_ascii(word_tokenize(sent)))
        
        sent= preprocess_to_str(sent)
        
        #print(list_sent)
        if sent!='':
            
            processed_text.append(sent)
            
        
        
    return processed_text

def process_sent(Text):
    processed_text=[]
    
    for i in range (len(Text.split('.'))):
        list_sent=list()
        sent=Text.split('.')[i].lower()
        sent=re.sub('[,\!?]', '.', sent)
        sent=' '.join(remove_non_ascii(word_tokenize(sent)))
        
        sent= preprocess_to_str(sent)
        
        #print(list_sent)
        if sent!='':
            list_sent.append(sent)
            processed_text.append(list_sent)
            
        
        
    return processed_text
def listToString(s):  
    
    # initialize an empty string 
    str1 = " " 
    
    # return string   
    return (str1.join(s)) 
        
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
def Tfidf(x,num_words):

    data=[]
    dic=[]
    raw=' '.join(pos_tagging(x))

    #print(raw)
    
    #print(type(dic))
    #from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    cv=TfidfVectorizer(min_df=0.6)
    word_count_vector=cv.fit_transform([raw])

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    feature_names=cv.get_feature_names()
        
    # get the document that we want to extract keywords from
    
    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([raw]))
    
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    
    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,num_words)
    
    #print("===Keywords===")
    #for k in keywords:
        #print("======")
        #print(k,keywords[k])
        #print (k)
      
    keywds=list(keywords.keys())
    #keywds
    return keywds


from nltk.corpus import wordnet as wn

from nltk.corpus import wordnet_ic
#Working on each document 
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth,fpmax,apriori #Implementation of frequent patterns algorithms to be applied on a set of texts within the same Topic
from nltk.tokenize import sent_tokenize, word_tokenize

def fpg(sent,support):
    x=dict()
    words=[]
    fpg = pd.DataFrame()
    for i in range(len(sent)):
        #print(sent[i])
        words.append((preprocess(sent[i])))
        #print(words)
    
    #print(classe)
    try : 
        te = TransactionEncoder()
        #patterns = pyfpgrowth. find_frequent_patterns(words, 10)
        #rules = pyfpgrowth. generate_association_rules(patterns,0.8)
        te_ary = te.fit(((words))).transform((words))
        words=[]

        df_r = pd.DataFrame(te_ary, columns=te.columns_)
        #print(df_r)
        #print(df_r)
        #print(rules)
        fpg = fpgrowth(df_r , min_support=support, use_colnames=True,max_len=15).sort_values(by='support' , ascending = False)#fpgrowth
    except ValueError :
        print('Value Error')
    print(fpg)
    return fpg
def resultat(Text):
    processed_Text_sent=process(Text)
    processed_Text=listToString(processed_Text_sent)
    lda_result=""
    lda_result=lda_t(processed_Text)
    lda_result=lda_result[0].split()
    tfidf_result=Tfidf(processed_Text,10)
    
    textRank=[]
    textRank=keywords.keywords(processed_Text).split('\n')
    textRank=textRank[:10]
    #print("lda= ",len(lda_result))
    #print("tfidf= ",len(tfidf_result))
    #print("textrank= ",len(textRank))
    lda_tfidf_textRank=[]
    lda_tfidf_textRank.append(lda_result)
    lda_tfidf_textRank.append(tfidf_result)
    lda_tfidf_textRank.append(textRank)
    Resultat_lda_tfidf_textRank=[]
    for i in range (len(lda_tfidf_textRank)):
        for j in range (len(lda_tfidf_textRank[i])):
            Resultat_lda_tfidf_textRank.append(lda_tfidf_textRank[i][j])
            
        
        
    
    return Resultat_lda_tfidf_textRank
def resultat_sans_textrank(Text):
    processed_Text_sent=process(Text)
    processed_Text=listToString(processed_Text_sent)
    lda_result=""
    lda_result=lda_t(processed_Text)
    lda_result=lda_result[0].split()
    tfidf_result=Tfidf(processed_Text,10)
    
    #print("lda= ",len(lda_result))
    #print("tfidf= ",len(tfidf_result))
    lda_tfidf=[]
    lda_tfidf.append(lda_result)
    lda_tfidf.append(tfidf_result)
    Resultat_lda_tfidf=[]
    for i in range (len(lda_tfidf)):
        for j in range (len(lda_tfidf[i])):
            Resultat_lda_tfidf.append(lda_tfidf[i][j])
            
        
        
    Resultat_lda_tfidf=' '.join(Resultat_lda_tfidf)
    return Resultat_lda_tfidf,tfidf_result
def resultat_textrank(Text):
    processed_Text_sent=process(Text)
    processed_Text=listToString(processed_Text_sent)
    
    
    textRank=[]
    textRank=keywords.keywords(processed_Text).split('\n')
    textRank=textRank[:10]
    
    return textRank

def resultat_fpg(Text):
    processed_Text_sent=process(Text)
         
        
        
    L=[list(x) for x in fpg(processed_Text_sent,0.1)['itemsets']]
    lf=[]
    for sublist in L:
        
        for item in sublist:
            
            lf.append(item)
    #lf=list(set(lf))
    fpg_result=lf
    if(len(fpg_result)<=10):
        print("0.07")
        L=[list(x) for x in fpg(processed_Text_sent,0.07)['itemsets']]
        lf=[]
        for sublist in L:
        
            for item in sublist:
            
                lf.append(item)
        #lf=list(set(lf))
        fpg_result=lf
    if(len(fpg_result)<=10):
        print("0.05")
        L=[list(x) for x in fpg(processed_Text_sent,0.05)['itemsets']]
        lf=[]
        for sublist in L:
        
            for item in sublist:
            
                lf.append(item)
        #lf=list(set(lf))
        fpg_result=lf
          
    if(len(fpg_result)<10):
        print("0.035")
        L=[list(x) for x in fpg(processed_Text_sent,0.035)['itemsets']]
        lf=[]
        for sublist in L:
        
            for item in sublist:
            
                lf.append(item)
        #lf=list(set(lf))
        fpg_result=lf
        
    print("fpg= ",len(fpg_result))
    res=fpg_result
    return res


#####
txt=str(sys.argv[1])

data,tfidf_result=resultat_sans_textrank(str(sys.argv[1]))
output1="%s" % (data)
print(str(output1))


id_user=int(sys.argv[2])
#print("id_user is ",id_user)
from context.models import Keywords,Contexte,document
from django.contrib.auth.models import User
#add new contexte
new_contexte=Contexte(etiquette=output1)
new_contexte.save()
#print(new_contexte.id)
#add new document
user= User.objects.get(pk=id_user)  
new_document=document(id_user=user,topic_id=new_contexte,Text=txt,Date=datetime.datetime.now())
new_document.save()
#add keywrds
for k in tfidf_result:
        new_keywords = Keywords(id_etiquette=new_contexte,mots=k)
        new_keywords.save()
