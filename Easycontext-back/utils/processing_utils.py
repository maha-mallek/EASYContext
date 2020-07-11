import os #For Folders management
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
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(' '.join(pos_tagging(text))):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    f_res= ' '.join(result)
    return result

 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    topics=[]
  
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        #print("\nTopic #%d:" % topic_idx)
        topics.append((" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])))
    return topics

def lda_c(text):

    count_vectorizer = CountVectorizer(stop_words='english')

    count_data = count_vectorizer.fit_transform([text])

    number_topics = 1
    number_words = 5

    lda_count = LDA(n_components=number_topics , doc_topic_prior=1, topic_word_prior=1, max_iter=2)

    lda_count.fit(count_data)
    x=print_topics(lda_count, count_vectorizer, number_words)
    return x

def lda_t(text):
    Tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    tfidf_data = Tfidf_vectorizer.fit_transform([text])
    number_topics = 1
    number_words = 10

    lda_tfidf = LDA(n_components=number_topics , doc_topic_prior=1, topic_word_prior=1, max_iter=2 , )

    lda_tfidf.fit(tfidf_data)

    y=print_topics(lda_tfidf, Tfidf_vectorizer, number_words)
    return y

def compare(tit,res):
    title = tit.split(' ')
    resultat = res[0].split(' ')
    x=0
  #print("Title is ",title)
  #print("resultat ",resultat)

    for t in title : 
        x+=resultat.count(t)
    #print("Title is {} , Resultat = {} , Score = {} ".format(title,resultat,(x/len(title))*100))
    return (x/len(title))*100

def split(text):
    
    l=text.split('.')
    
    ll=list()
    for i in range(len(l)):
        """if (l[i].count('.')>1) or (not l[i].endswith('.')) :
            l[i] = l[i].replace('.','')"""
        #sent_l = " ".join(l)
        sent_l=l[i]
        sent_l= sent_l.replace("_"," ")
        sent_l= sent_l.replace("   "," ")
        sent_l = sent_l.replace(".","_")
        sent_l = re.sub("[^a-zA-Z0-9.\s]",' ',sent_l)
        sent_l = re.sub("perform",' ',sent_l)

        sent_l = re.sub(r'[\s]+',' ',sent_l)
        #sent_l = re.sub(r'\b\w{1,3}\b','',sent_l)
        sent_l = sent_l.replace('_',".")
        #print("sentence : ",sent_l)
        
        if (sent_l!='')&(sent_l!=' '):
            ll.append(sent_l)  
    for y in ll:
        if y=='  'or y==' 'or y=='': ll.remove(y)
        
    return(ll)
from nltk.corpus import wordnet as wn

from nltk.corpus import wordnet_ic
#Working on each document 
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth,fpmax,apriori #Implementation of frequent patterns algorithms to be applied on a set of texts within the same Topic
from nltk.tokenize import sent_tokenize, word_tokenize

def fpg(sent):
    x=dict()
    words=[]
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
        fpg = fpgrowth(df_r , min_support=0.1, use_colnames=True,max_len=3)#fpgrowth
    except ValueError :
        print('Value Error')
    print(fpg)
    return fpg
def jcn_process(xfpg,keywds):
    #fpg_l=list(set1)
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    x=[]
    s_g=[]
    result=[]
    #print (len(l))
    #print (len(l2))
    #for k in list(keywds):
    #   for j in list(set1):
    #        s1 = wn.synsets(k)
    #        s2 = wn.synsets(j)
    #       print (k,j)
    #      x.append(wn.jcn_similarity(s1[i], s2[i], brown_ic))
    #print(len(set1))
    print("TFIDF KEYWORDS NUMBER "+str(len(keywds)))

    for j in ( list(xfpg['itemsets'])):
        s=[]
        result=[]
        #print()
        #print(j)
        #print(len(j))
        #print()
        
        #print("FPGROWRH ITEMSET length "+str(len(list(xfpg['itemsets']))))  
        for j1 in j:
            #print("FPGROWRH ITEMSET  "+str((j1)))  
            for i in (keywds[:20]):
                

                s1 = wn.synsets(i)
                s2 = wn.synsets(j1)
            
            #print (s1,s2)
            #print(s1)
            #print(s2)
            
                if(len(s1) == 0 or len(s2)==0 ):
                    x=0
                elif (s1[0].pos() != s2[0].pos()) :
                    x=0
                else:
                    x=wn.jcn_similarity(s1[0],s2[0], brown_ic)
                    if (x==(1e300)):
                        x=1
                    result.append(x)
                
                #print ("jcn of {} and {} is {}".format(str(j1),str(i),str(x)))
            s1=sum(result)
            #print("score of {} is {}".format(j1,str(s1)))
        s.append(s1)
    
        #print("score of {} is {}".format(str(j),str(sum(s))))
    #print(result)
        s_g.append(s)

            #print("Length of result"+str(len(s_g)))

    #res=s/len(result)
    #print("resultat" + str(s_g))

    #print("Resultat" + str(res*100))
    res=np.asarray(s_g)
    #get the element with the higest score 
    if(res.size != 0):
        res_index=res.argmax()
        print("Highest score is {} with {}".format(str(xfpg['itemsets'][res_index]),str(s_g[res_index])))
         #Converting frozen set to list
        sets=[xfpg['itemsets'][res_index]]

        final_list = ([list(x) for x in sets])

        return      final_list
    
    else:
        None
    #res_index
    print()
    return None
    
    print()

import nltk
from sklearn.feature_extraction.text import TfidfTransformer
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

    print(raw)
    
    print(type(dic))
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
    print("\n===Keywords===")
    for k in keywords:
        print(k,keywords[k])
        #print (k)
    keywds=list(keywords.keys())
    #keywds
    return keywds

def compare_jcn_title(df1,tit,res,i):
    
    title = tit.split(' ')
    resultat=[]
    try:
        
        resultat = list(res[0])[0]
    except: 
        # catch *all* exceptions
        
        e = sys.exc_info()[0]
        print(e)
    
    x=0
    #print("title elements ",title)
    #print("jcn elements ",resultat)

    for t in title : 
        x+=resultat.count(t)
        #print(x)
    
    print("{} and {} is {} , Score : {}".format(title,resultat,x,(x/len(title))*100))

    return (x/len(title))*100    
