from collections import defaultdict,OrderedDict,Counter
import math
import numpy as np
import os
import string
import re
import sys
import zlib
import nltk
import bs4
    
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt',quiet=True)
nltk.download('averaged_perceptron_tagger',quiet=True)

nltk.download('stopwords',quiet=True)



###INIT
number=re.compile(r'^\d+$')
punctuations = (string.punctuation+'&’\n').replace("'", '')

ps = PorterStemmer()


def preprocesser(input_data_list, min_len=1):
    
    tagged_data=Counter()
    data=''
    for input_data in input_data_list:
        #POL erion
        for tag in ['person','organization','location']:
            for x in input_data.find_all(tag):
                x=x.text.strip()
                if x:
                    tagged_data[tag[0]+':'+x.lower()]+=1
                    for y in x.split():
                        tagged_data[tag[0]+':'+y.lower()]+=1

        data+=' '+input_data.text

    #punctuation removal and double space removal
    for punc in punctuations:
        data=data.replace(punc,' ')#'' or ' '
        for tag in 'plo':
            if punc+':'+tag in tagged_data:
                del tagged_data[punc+':'+tag]
    
    data=data.replace('  ',' ').lower()

    #tokenization
    data=Counter(nltk.word_tokenize(data))
    
    

    #lemmatization
    # data=lemmatizer(data)


    #number to words
    # data = map((lambda x: num2words(x) if number.fullmatch(x) else x), data)

    #stopwords removal
    stopword_list=[tag+word for word in stopwords.words('english') for tag in ['','o:','l:','n:']]
    
    # data=[x for x in data if x not in stopword_list and len(x)>min_len]
    data={k:v for k,v in data.items() if k not in stopword_list and len(k)>min_len}

    #stemming    
    # data=list(map(ps.stem,data))
    data={ps.stem(k):v for k,v in data.items() }

    return {**data , **tagged_data}   



def extracter(file): 
    
    data=''
    with open(file,'r+') as tmp:
        data= tmp.read()
    #to fuse all the adacent tagged words
    for tag in ['</PERSON> <PERSON>','</ORGANIZATION> <ORGANIZATION>','</LOCATION> <LOCATION>',
                '</PERSON><PERSON>','</ORGANIZATION><ORGANIZATION>','</LOCATION><LOCATION>','-rrb-','-lrb-']:
        data = data.replace(tag,'')


    f=bs4.BeautifulSoup(data,'lxml')
    d ={}
    for doc in f.find_all('doc'):
        if doc.find('text').text is not '\n':
            d[doc.docno.text.strip()]=doc.find_all('text')
    return d



def generate_inverted_index(path_dir):
    
    allfile=[t.path for t in os.scandir(path_dir)]

    inverted_list = defaultdict(lambda: defaultdict(lambda:0))
    doc_embed={}
    doc_embed_id=0
    for file in allfile:
        docs = extracter(file)
        for docid,single_doc in docs.items():
            if docid not in doc_embed:
                doc_embed[docid] = doc_embed_id
                doc_embed_id += 1
            single_doc=preprocesser(single_doc)
            for term, tf in single_doc.items():
                inverted_list[term][doc_embed[docid]]=tf

    embed_to_doc={v:[k,0] for k,v in doc_embed.items()}
    N= len(embed_to_doc)
    vocab={}                                                  
    ''' make it more efficient'''
    for term, doc_list in inverted_list.items():
        s=''
        df=len(doc_list)
        vocab[term]={'df':df}
        for em_docid, tf in doc_list.items():
            embed_to_doc[em_docid][1]+=((1+math.log(tf))*math.log(1+N/df))**2
            s+=';' + str(em_docid) + ':' + str(tf)
        vocab[term]['pststr']=s[1:]
    embed_str=''
    for k,v in embed_to_doc.items():
        embed_str+='?'+str(k)+'%'+v[0]+'%'+str(v[1]**0.5)

    return      OrderedDict(sorted(vocab.items())), embed_str[1:]



def write_dict_to_file(vocab, doc_embed, dict_file, idx_file):

    offset=0
    vocab_str=''
    with open(idx_file,'w+b') as f:
        for k,v in vocab.items():
            x = zlib.compress(v['pststr'].encode('utf-8'))
            l = len(x)#sys.getsizeof
            vocab_str+='?'+k+'%'+str(v['df'])+'%'+str(l)+'%'+str(offset)#'start':offset,'len':l,'df':v['df']}
            offset+=l
            f.write(x)
    
    d = vocab_str[1:]+'='+doc_embed
    # word%df%len%start?................=docembed%docid%normalisationfactor
    with open(dict_file,'w+b') as f:
        f.write(zlib.compress(d.encode('utf-8')))


def train(corpus_path,dict_file,idx_file):
    
    vocab, doc_embed = generate_inverted_index(corpus_path)
    write_dict_to_file(vocab, doc_embed, dict_file, idx_file)
    

if __name__=='__main__':

    train(sys.argv[1],os.path.join(os.getcwd(),sys.argv[2]+'.dict'),os.path.join(os.getcwd(),sys.argv[2]+'.index'))