import zlib
from collections import defaultdict,OrderedDict
import re
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
nltk.download('stopwords',quiet=True)
import math

ps=PorterStemmer()
def preprocess(filepath, vocab, doc_embed):
    
    s=open(filepath,'r+').read()

    topic,number= re.compile('<title> *Topic\:.*?\n'), re.compile('<num> *Number\:.*?\n')
    num2=re.compile('[1-9][0-9]*')
    top2=re.compile('Topic\:.*?\n')
    queries={int(num2.findall(k)[0]):top2.findall(v)[0][6:].strip() for k,v in zip(number.findall(s),topic.findall(s))}

    for k,v in queries.items():
        queries[k]=tokenize( v, vocab, doc_embed)
    return queries

def tokenize(query, vocab, doc_embed):
    
    finalqueries=[]
    tagged,prefix,simple=[],[],[]

    # stopwords removal
    for i in [ x for x in (string.punctuation+'&’\n') if x not in "':*"]:
        query.replace(i,' ')


    for q in query.split():
        if q and q[-1]=='*':
            if len(q)>2 and q[:2]=='n:':
                prefix+=[tg+q[1:] for tg in 'plo']
            else: prefix.append(q.lower())

        elif len(q)>=2 and q[1]==':':
            if q[0]=='n':
                tagged+=[tg+q[1:] for tg in 'plo']
            else: tagged.append(q)
        else: simple.append(q)
    
    for p in prefix:
        finalqueries+=re.compile(p).match(vocab.keys())

    finalqueries+=[term for term in tagged if term in vocab]
    ###process normal query

    simple=list(map(ps.stem,simple))

    finalqueries+=[word for word in simple if word in vocab]
    


    score=defaultdict(lambda: 0)
    for term in finalqueries:
        score[term]+=1
    n,N=0,len(doc_embed)
    for term,tf in score.items():
        score[term]=(1+math.log(tf))*math.log(1+N/vocab[term]['df'])
        n+=score[term]**2
    
    n=n**0.5

    for term,sc in score.items():
        score[term]=sc/n

    return score   
     

def retrive_qrels(query_filepath,vocab_file, idx_file,output_filepath ,k):
    
    vocab, doc_embed = extract(vocab_file)
    
    query_tokens_dict = preprocess(query_filepath,vocab,doc_embed)
    
    with open(idx_file, 'r+b') as f:
        with open(output_filepath, 'w+') as g:
            for query_no,query_tokens in query_tokens_dict.items():
                docDict=top_k(query_tokens,vocab,f,doc_embed,k)
                if k>len(docDict): 
                    print('cutoff too large')
                    continue
                for doc in list(docDict.keys())[:k]:
                    g.write(str(query_no)+' 0 '+doc_embed[doc][0]+' 0 '+str(docDict[doc])+' r\n\n')

    
        




def top_k(query_tokens,vocab,f,doc_embed,k):
    
    N=len(doc_embed)

    score=defaultdict(lambda:0)
    
    for term,term_score in query_tokens.items():
        score=get_score(term,term_score,vocab,f,N,doc_embed,score)
    
    topDoc=OrderedDict(sorted(score.items(),key=lambda kv: kv[1], reverse=True))
    
    if len(topDoc)<k:
        for key in doc_embed.keys():
            if key not in topDoc:
                topDoc[key]=0
            if len(topDoc)>=k: break

    return topDoc

def get_score(term,term_score,vocab,f,N,doc_embed,pst_lst = defaultdict(lambda:0)):
    f.seek(vocab[term]['start'])
    
    for x in zlib.decompress(f.read(vocab[term]['len'])).decode("utf-8").split(';'):
        a,b = x.split(':')
        pst_lst[a] += (1+math.log(int(b)))*math.log(1+N/vocab[term]['df'])*term_score/(doc_embed[a][1])
    
    return pst_lst

def extract(vocab_file):
    
    with open(vocab_file, 'r+b') as f:
        fi=zlib.decompress(f.read()).decode('utf-8').split('=')#'iso-8859-1'
        vocab,doc_embed={},{}
        for oneterm in fi[0].split('?'):
            l=oneterm.split('%')
            vocab[l[0]]={'df':int(l[1]), 'len':int(l[2]),'start' :int(l[3])}
        for oneterm in fi[1].split('?'):
            l=oneterm.split('%')
            doc_embed[l[0]]=[l[1],float(l[2])]
            
    return vocab,doc_embed

if __name__=='__main__':
    nargs={}
    for i in range(1,len(sys.argv),2):
        nargs[sys.argv[i]]=sys.argv[i+1]
    nargs['--cutoff']=nargs.get('--cutoff',10)
    
    retrive_qrels(nargs['--query'],nargs['--dict'], nargs['--index'], nargs['--output'],nargs['--cutoff'])

