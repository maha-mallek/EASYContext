import scipy
import os
import requests
import sys
import utils.processing_utils as up 
#Django SIDE
from django.shortcuts import render
import json
from django.shortcuts import render
from subprocess import run,PIPE
from django.http import HttpResponse
from django.utils.translation import gettext
from .models import Contexte, Keywords,document
from .serializers import ContexteSerializer, KeywordSerializer,DocumentSerializer
from django.http import HttpResponse
from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser 
from rest_framework import status
from django.contrib.auth.models import User
######################## Application SIDE
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from summa import keywords
import sys
import datetime
import gensim #For Text processing
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
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
#Preparing necessities : 
df = pd.read_csv('database_v1.csv')
etiquettes = df['Semantic Similarity FPG and LDA'].values.tolist()
textes = df['raw'].values.tolist()
keywords1 = df['keywords'].values.tolist()
df['Title'] = df['Class/Textname'].map(lambda x : x.split('/')[1])
title = df['Title'].values.tolist()
#for preprocessing
def preprocess_to_str(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    f_res= ' '.join(result)
    return f_res

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return (new_words)
import nltk
def lemmatize_stemming(text):
    ps = PorterStemmer() 
    lem = WordNetLemmatizer().lemmatize(text)
    stem = ps.stem(lem)
    return lem
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
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(' '.join(pos_tagging(text))):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    f_res= ' '.join(result)
    return result
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

#LDA
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
def print_topics(model, count_vectorizer, n_top_words):
    topics=[]
    
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        #print("\nTopic #%d:" % topic_idx)
        topics.append((" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])))
        #print(topics)
    return ' '.join(topics)

def lda_t(text):
    Tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    tfidf_data = Tfidf_vectorizer.fit_transform([text])
    number_topics = 1
    number_words = 10 
    

    lda_tfidf = LDA(n_components=number_topics , doc_topic_prior=1, topic_word_prior=1, max_iter=2 , )

    lda_tfidf.fit(tfidf_data)

    y=print_topics(lda_tfidf, Tfidf_vectorizer, number_words)
    return y
 
 
 #fp-GROWTH
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth,fpmax,apriori #Implementation of frequent patterns algorithms to be applied on a set of texts within the same Topic
def fpg(sent,support):
    x=dict()
    words=[]
    fpg = pd.DataFrame()
    for i in range(len(sent)):
        #print(sent[i])
        words.append((preprocess(sent[i])))
    try : 
        te = TransactionEncoder()
        te_ary = te.fit(((words))).transform((words))
        words=[]
        df_r = pd.DataFrame(te_ary, columns=te.columns_)
        fpg = fpgrowth(df_r , min_support=support, use_colnames=True,max_len=3).sort_values(by='support' , ascending = False)#fpgrowth
    except ValueError :
        print('Value Error')
    print(fpg)
    return fpg

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
    return ' '.join(res)
#TF-Universal
#Load Universal encoder 
import tensorflow_hub as hub
import matplotlib.pyplot as plt

module_url = "4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model1 = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
    return model1(input)
import pandas as pd
def plot_similarity(labels, features, rotation):


    corr = np.inner(features, features)
    sns.set(font_scale=1.2)
    g = sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")

from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
def run_and_plot(fpg_input , others):
    #print("fpg input in run and plot is {}".format(fpg_input))
    #
    # print("lda input in run and plot is {}".format(others))

    #This is the input for fpg items
    print("*run*")
    fpg_input = fpg_input.split(' ')
    others = others.split(' ')
    fpg_embeddings = embed(fpg_input)
    print(fpg_input)

    #TF-IDF TEXTRANK AND LDA EMBEDDINGS
    others_embeddings = embed(others)
    #Calculating similiarty correlation
    corr = np.inner(fpg_embeddings,others_embeddings)
    #print(corr)
    #final= pd.DataFrame(np.random.random(size=(len(corr))))
    final = pd.DataFrame(corr ,  fpg_input , others)
    #print(final)
    #plot_similarity(messages_, message_embeddings_, 90)
    final_ = pd.DataFrame(final.sum(axis=1 , skipna =True).sort_values(ascending = False) , columns=[1] , dtype=str)
    
    
    return  final_

#Function for processing similarity output
def processing_similarity(sim):
  x = (re.sub('[\d]','',(str(sim))))
  x = re.sub(' ','',x)
  x = re.sub('[.\n]', '\n', x)
  x = re.sub('[\s]','\n',x)
  x = word_tokenize(x) #trajaa list
  return x
  
  ###Semantic similarity with BERT
""" from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens') """

def sbert_semantics(query,queries): 
    #queries houma les context li fl base
    #query heya l bsh naamlou beha recherche
    query = [query]
    queries_embeddings = model.encode(queries)
    query_embedding = model.encode(query)
    results=[]
    # Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
    number_top_matches = 3 #@param {type: "number"}
    
    print("Semantic Search Results")

    for query, query_embedding in zip(query, query_embedding):

        distances = scipy.spatial.distance.cdist([query_embedding], queries_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        #besh tedhhak alia :)hhhh aya khademha bark 
        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")
        saved_index = []
        for idx, distance in results[:number_top_matches]:
            print(queries[idx].strip(), "(Cosine Score: %.4f)" % (1-distance))
            #list of id
            saved_index.append(idx)

        saved_index[0] += 1 #tsir haka?fibeli tsir
        return saved_index[0],queries[results[0][0]],(1-results[0][1])

#with tf hub semantics
def tfhub_semantics(query,queries): 
    #queries houma les context li fl base
    #query heya l bsh naamlou beha recherche
    query = [query]
    queries_embeddings = embed(queries)
    query_embedding = embed(query)
    results=[]
    # Find the closest 3 sentences ofthe corpus for each query sentence based on cosine similarity
    number_top_matches = 3 #@param {type: "number"}
    
    print("Semantic Search Results")

    for query, query_embedding in zip(query, query_embedding):

        distances = scipy.spatial.distance.cdist([query_embedding], queries_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        #besh tedhhak alia :)hhhh aya khademha bark 
        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")
        saved_index = []
        for idx, distance in results[:number_top_matches]:
            print(queries[idx].strip(), "(Cosine Score: %.4f)" % (1-distance))
            #list of id
            saved_index.append(idx)

        saved_index[0] += 1 #tsir haka?fibeli tsir
        return saved_index[0],queries[results[0][0]],(1-results[0][1])
# Create your views here.


def external(request):
    return render(request,'contexte.html')

def result(request):
    inp= request.POST.get('text')
    direct=os.getcwd()
    #up is processing utils 
    inp_proc = up.preprocess_to_str(inp)
    etiquette  = up.lda_t(inp)
    keywords , score , affichage_keywords = up.Tfidf(inp_proc,5)
    fpg = "fpg output"
    text_sent = up.split(inp)
    fpg = up.fpg(text_sent)
    res="None"
    #res = up.jcn_process(fpg,out[0].split(' '))
    out  = "Xx"
    #out= run([sys.executable,direct+'//summarize.py',inp],shell=False,stdout=PIPE)
    #data1="Hello Jawhar"
    return render(request ,'contexte.html',{'Keywords':str((out)),'Fpg':str((fpg)),'Result':res,'Etiquette':etiquette,'Keys':keywords,'Scores':score})

@csrf_exempt
def add(request):
  
  if request.method == 'POST':
      #req = JSONParser().parse(request)
      #print(req)
      #inp= req['text']
      inp= request.POST.get('text')

      inp_proc = up.preprocess_to_str(inp)

      etiquette = ' '.join(up.lda_c(inp_proc)) #We'll use the lda result as an etiquette 
      keywords , score , affichage_keywords = up.Tfidf(inp_proc,5)
      #Intiliizaing curent current contexte
      context = Contexte(etiquette = etiquette )
      context.save() #Saving to the database
      #Intiliazing the keywords linked via the id of the context
      keys = Keywords(id_contexte = Contexte.objects.filter(etiquette = context).order_by("-pk")[0] , mots =' '.join(list(keywords)) , score = ' '.join(list(score)))
      keys.save()
      context_serialized = ContexteSerializer(context , many = True)
      keywords_serialized = KeywordSerializer(keys , many=True)

      return JsonResponse(keywords_serialized.data, safe=False)



from django.contrib.auth.models import User



""" from sentence_transformers import SentenceTransformer
 """
###################New
@csrf_exempt
def context(request):
    
    return render(request,'context2.html')

@csrf_exempt
def lda(request):

    if request.method == 'POST':
        text_data = JSONParser().parse(request)
      
        inp=text_data['text']
        print("1*")
        #print(inp)
        #inp= request.POST.get('texte')
        id_user=str(text_data['id_user'])
        direct=os.getcwd()
        #out= run([sys.executable,direct+'//context.py',inp,id_user],shell=True,stdout=PIPE)
        txt=str(inp)
        fp_growth = resultat_fpg(txt)
        #data,tfidf_result=resultat_sans_textrank(str(sys.argv[1]))
        fpg_output="%s" % (fp_growth)
        #print(str(fpg_output))
        lda_output = lda_t(txt)
        lda_output="%s" % (lda_output)
        print("2*")
        print(lda_output)
        sim = processing_similarity(run_and_plot(fpg_output,lda_output))
        res = [] 
        [res.append(x) for x in sim if x not in res] 
        res_sim=res[:10] 
        print("3*")
        sim_p =' '.join((res_sim[:5])) #####################3
        
         
        id_context,resultat,flag = tfhub_semantics(str(lda_output),keywords1)


        id_user=int(id_user)
        #print("id_user is ",id_user)
        user= User.objects.get(pk=id_user)

        #Save new document
        #new_document=document(id_user=user,topic_id=new_contexte,Text=txt,Date=datetime.datetime.now())
        #new_document.save()
        print("nearest words are {} with ditance {}".format(resultat,flag))
        if flag < 0.7:
            
            user= User.objects.get(pk=id_user) 
            
            new_contexte=Contexte(etiquette=sim_p) #etiquette finale
            
            new_contexte.save()
            new_document=document(id_user=user,topic_id=new_contexte,Text=txt,Date=datetime.datetime.now())
            new_document.save() 
            #add keywrds
            for token in lda_output.split(' '):
                new_keywords = Keywords(id_etiquette=new_contexte,mots=token)

                new_keywords.save()
            try:
                context = Contexte.objects.filter().order_by("-pk")[0]  #objects.order_by('id')[0] #
                id_c=context.id
            except Contexte.DoesNotExist: 
                return HttpResponse(status=status.HTTP_404_NOT_FOUND) 

            contextes = Contexte.objects.all()
            #last id 
            for res in contextes:
                Id=res.id
                LastId=Id

            print(LastId)
            #l=len(contextes)

            #print(l)
            contexte = Contexte.objects.filter(pk=5)[0]
            
            context_serialized = ContexteSerializer(context)   
            print(" the contexet is ===================== ",context_serialized.data)
            keys = Keywords.objects.filter(id_etiquette = id_c )
    
            keywords_serialized = KeywordSerializer(keys , many=True)
            #JsonResponse(context_serialized.data,safe=False),
            #JsonResponse(keywords_serialized.data,safe=False)
            json_result = json.dumps({'context': context_serialized.data, 'keywords': keywords_serialized.data})
            return HttpResponse(json_result)
        else:

            print("else")
            #add new keywords to current keyword
            #getting current keywords and context
            print(id_context)
            context_old = Contexte.objects.get(pk=id_context) 
            new_document=document(id_user=user,topic_id=context_old,Text=txt,Date=datetime.datetime.now())
            new_document.save() 
            keys = Keywords.objects.filter(id_etiquette = context_old)

            #keywords = Keywords.objects.values_list('mots', flat=True).get(id_etiquette=context) 
            #keywords = Keywords.objects.filter(id_etiquette= context) 
            #print("keywords bef update : {}".format(keywords))
            #adding new keywords to database
            #keywords += resultat
            #print("keywords afeter update : {}".format(keywords))
            context_serialized = ContexteSerializer(context_old)

            print("========================= context with else",context_serialized.data)
            keywords_serialized = KeywordSerializer(keys , many=True)
            #JsonResponse(context_serialized.data,safe=False),
            #JsonResponse(keywords_serialized.data,safe=False)
            json_result = json.dumps({'context': context_serialized.data, 'keywords': keywords_serialized.data})
            return HttpResponse(json_result)


## for download recent results
@csrf_exempt 
def recent_result(request, pk):
    try: 
        user = User.objects.get(pk=pk) 
    except User.DoesNotExist: 
        return HttpResponse(status=status.HTTP_404_NOT_FOUND) 
 
    if request.method == 'GET': 
        try: 
            doc = document.objects.filter(id_user=pk) #get - filter bch temchy barcha res ama erreuret
            
        except User.DoesNotExist: 
            return HttpResponse(status=status.HTTP_404_NOT_FOUND)
        
        #print(doc)#document object
        #print(doc.topic_id)#contexte object
        #id_context = document.objects.values_list('topic_id',flat=True)[10]
        #print(id_context)
        #context_serializer = ContexteSerializer(doc.topic_id)
        document_serializer = DocumentSerializer(doc, many=True) 
        
        #id_c=document_serializer.data['topic_id']
        
        #json_result = json.dumps({'document': document_serializer.data, 'contexte': context_serializer.data})
       
        return JsonResponse(document_serializer.data,safe=False) 
        #return HttpResponse(json_result,status=status.HTTP_200_OK) #,safe=False
 
@csrf_exempt 
def keywords_result(request, pk):

    context=Contexte.objects.get(pk=pk)
    #print(context)
    context_serializer = ContexteSerializer(context)

    keys = Keywords.objects.filter(id_etiquette = pk )
        
    keywords_serializer = KeywordSerializer(keys, many=True)
    #print(keywords_serializer.data)
    #return JsonResponse(keywords_serializer.data,safe=False) 
    json_result = json.dumps({'keys': keywords_serializer.data, 'contexte': context_serializer.data})
    return HttpResponse(json_result,status=status.HTTP_200_OK)


   

#############################

from .models import * 
import pandas as pd 
import django
import os 
from django.contrib.auth.models import User
import datetime
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()




user= User.objects.get(pk=1)
#To full up the database execute this cell by cell
#cell1
#for t in title : 
#    print(t)
#    contexte = Contexte(etiquette = t)
#    contexte.save() 
"""
#for context in etiquettes:
    #print(context)
    #contexte = Contexte(etiquette=context)
   # contexte.save() """

 

#cell2
#i=1
#for text in textes:
#    print('text',str(i))
#    c = Contexte.objects.get(pk=i)
#    texte = document(id_user=user,topic_id = c  ,Text = text , Date=datetime.datetime.now())
#    texte.save()
#    print('text {} saved'.format(str(i)))

#    i+=1
 
#cell3
#i=1
#for keys in keywords1:
#    for k in keys.split(' '):

#        print(k)
#        c = Contexte.objects.get(pk=i)
#        key_b = Keywords(id_etiquette= c,mots = k)
#        key_b.save()
#    i=i+1  

######################################################


""" from sentence_transformers import SentenceTransformer

# Load the BERT model. Various models trained on Natural Language Inference (NLI) https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/nli-models.md and 
# Semantic Textual Similarity are available https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/sts-models.md

model = SentenceTransformer('bert-base-nli-mean-tokens')
 """