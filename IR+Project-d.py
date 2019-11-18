
# coding: utf-8

# In[1]:


from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import os
import re
import math
from scipy import spatial
import numpy as np
import pprint
import sklearn.cluster
from sklearn.cluster import spectral_clustering
import pickle


def SpecialCharacters(word):
	if len(word)<3:
		return 1
	return 0


# In[2]:


inp_stop=open('data/stopwords.txt','r+') #read stopword from file
stopwords=eval(inp_stop.read())
print stopwords
id=-1
i=0
directory="d"
II={}
Document_TF_Count={}


# In[3]:


Documents=os.listdir(directory)
for Document in Documents:
	if i<4:
		i=i+1
	else:
		break
	f=open(directory+"/"+Document,'r');
	c=f.read()
# 	c=f.read().split('\n')
# 	print c
	f.close();
	# id=c[4].replace('DOC ID:','')
	id=id+1
	content=c
# 	content=c[5].replace('CONTENT:','')
	print content
	words=word_tokenize(str(content))
	lower_words=[x.lower() for x in words if not SpecialCharacters(x)]
	upper_words=[x for x in words if x.upper()==x and not SpecialCharacters(x)]
	words=[]
	words=lower_words+upper_words
	WordCount1={}
	#	
	for w in words:
		# if word in stopwords:
		#Term Frequency 
		if w in WordCount1:
			WordCount1[w]=WordCount1[w]+1
		else:
			WordCount1[w]=1
		#Inverted Index	
		if w in II:
			if id not in II[w]:
				II[w].append(id)
		else:
			II[w]=[]
			II[w].append(id)
	Document_TF_Count[id]=WordCount1


# In[4]:


#Assign Term Frequency as Zero
vocab=II.keys()
for v in vocab:
	for key in Document_TF_Count:
		if v not in Document_TF_Count[key]:
			Document_TF_Count[key][v]=0
print Document_TF_Count


# In[5]:


##Stemming#
stemmer=PorterStemmer()
l=len(vocab)
for key in Document_TF_Count:
	for t1 in range(l):
		for t2 in range(t1+1,l):
			v1=vocab[t1]
			v2=vocab[t2]
			stem1=stemmer.stem(v1)
			stem2=stemmer.stem(v2)

			tc1=Document_TF_Count[key][v1]
			tc2=Document_TF_Count[key][v2]	
			if tc1>0 and tc2>0 and v1!=v2 and stem1==stem2 and v1.upper()!=v2 and v2.upper()!=v1:
			#terms should exists in the same document not be equal, 
			#even upper or lower case should not be equal but stem should be			
				# print stem1
				# print stem2
				Document_TF_Count[key][v1]=tc1+1
				Document_TF_Count[key][v2]=tc1+1
print Document_TF_Count
#Calculate TF-IDF
print Document_TF_Count.keys()


# In[6]:


with open('Document_TF_Count', 'wb') as fp:
    pickle.dump(Document_TF_Count, fp)

with open ('Document_TF_Count', 'rb') as fp:
    Document_TF_Count = pickle.load(fp)


# In[7]:


term_vector={}
for v in vocab:
	idf=math.log((float(id+1)/len(II[v])),10)
	array=[]
	for key in Document_TF_Count:
		k=int(key)
		array+=[Document_TF_Count[key][v]*idf]
	term_vector[v]=array
print "Term Vector without sumation similarity"
pprint.pprint(term_vector)
#~ print "++++++++++++++++++"


# In[8]:


with open('termvector', 'wb') as fp:
    pickle.dump(term_vector, fp)

with open ('termvector', 'rb') as fp:
    term_vector = pickle.load(fp)


# In[9]:


# Update TF-IDF using TMI
def sim_f(word1, word2):
	if word1==word2:
		return 0
	max_value=0.0
# 	wordFromList1 = wordnet.synsets(word1)
# 	wordFromList2 = wordnet.synsets(word2)
# 	for wl1 in wordFromList1:
# 		if wl1 in wordFromList2:
# 			return 1
	return 0	
# 		for wl2 in wordFromList2:
# 			p=wl1.wup_similarity(wl2)
# 			if max_value < p: 
# 				max_value = p
# 	return max_value

def sim_f1(word1, word2):
	if word1==word2:
		return 0
	max_value=0.0
	wordFromList1 = wordnet.synsets(word1)
	wordFromList2 = wordnet.synsets(word2)
	for wl1 in wordFromList1:
		if wl1 in wordFromList2:
			return 1
	return 0	
# 		for wl2 in wordFromList2:
# 			p=wl1.wup_similarity(wl2)
# 			if max_value < p: 
# 				max_value = p
# 	return max_value

def sim_f2(word1, word2):
	if word1==word2:
		return 0
	max_value=0.0
	wordFromList1 = wordnet.synsets(word1)
	wordFromList2 = wordnet.synsets(word2)
	for wl1 in wordFromList1:
# 		if wl1 in wordFromList2:
# 			return 1
# 	return 0	
		for wl2 in wordFromList2:
			p=wl1.wup_similarity(wl2)
			if max_value < p: 
				max_value = p
	return max_value*0.1


# In[10]:


sim={}
for term1 in term_vector:
	sim[term1]={}
	for term2 in term_vector:
		sim[term1][term2]=sim_f2(term1,term2)

with open('sim', 'wb') as fp:
    pickle.dump(sim, fp)

with open ('sim', 'rb') as fp:
    sim = pickle.load(fp)
print "Similirity",sim


# In[11]:


for term in term_vector:
 	l=len(term_vector[term])
 	#~ print term
 	for i in range(l):
		sumation=0
		for v in vocab:
			sumation+=term_vector[v][i]*sim[term][v]	
		term_vector[term][i]+=sumation		
print "Term Vector sumation similarity"
pprint.pprint(term_vector)


# In[12]:


Matrix={}
for tv1 in term_vector:
	Matrix[tv1]={}
	for tv2 in term_vector:
		Matrix[tv1][tv2] =1-spatial.distance.cosine(term_vector[tv1],term_vector[tv2])
        
Distance_Matrix=np.array(Matrix)
print "Distance Matrix"
pprint.pprint(Distance_Matrix)


# In[13]:


with open('matrix', 'wb') as fp:
    pickle.dump(Matrix, fp)

with open ('matrix', 'rb') as fp:
    Matrix = pickle.load(fp)


# In[14]:


#~ from sklearn.cluster import AgglomerativeClustering
#~ agg = AgglomerativeClustering(n_clusters=5, affinity='precomputed')
#~ agg.fit_predict(Distance_Matrix)
i=0
l=len(vocab)
w, h = 8, 5;
DM = [[0 for x in range(l)] for y in range(l)]
labels={}

i=0
for l1 in Matrix:
	labels[i]=l1
	j=0
	for l2 in Matrix[l1]:
		if Matrix[l1][l2]<0.5:
			DM[i][j]=0            
		else:
			DM[i][j]=Matrix[l1][l2]
		j=j+1
	i=i+1	
vocab=np.asarray(vocab)	

print DM
print "Vocab",vocab
print "Labels",labels


# In[15]:


#MCL Start	
import mcl_c
import networkx as nx
DM=np.array(DM)
G = nx.from_numpy_matrix(np.matrix(DM))
#~ print "============"
#~ print DM
#~ print "============"
#~ print G.nodes()
#~ print "============"
#~ print labels
#~ #inflate > more finer


# In[20]:


M, clusters = mcl_c.mcl(DM, expand_factor = 5, inflate_factor = 2, max_loop = 10, mult_factor = 5)
print "Clusters",clusters
pos=mcl_c.draw(G,M,clusters,labels)
#MCL END


# In[21]:


for k in clusters:
    t=[]
    for v in clusters[k]:
        t+=[labels[v]]
    if len(t)==len(clusters[k]):
        for i in t:
            final[no]=t
            no=no+1
print final    


# In[22]:


keywords=['Education','Technology']

for l in final:
    keypoint=[]
    for k in keywords:
        point=0
        wordFromList1 = wordnet.synsets(k)
        for i in final[l]:
            wordFromList2 = wordnet.synsets(i)
            p=0
            for wl1 in wordFromList1:
                for wl2 in wordFromList2:
                    sim_p=wl1.wup_similarity(wl2)
                    if sim_p>p:
                        p=sim_p
            point=point+p
        keypoint.append(point)
    max_value = max(keypoint)
    max_index = keypoint.index(max_value)
    print final[l],keywords[max_index]
    


# In[17]:


M, clusters = mcl_c.mcl(DM, expand_factor = 4, inflate_factor = 2, max_loop = 10, mult_factor = 6)
print "Clusters",clusters
pos=mcl_c.draw(G,M,clusters,labels)
print len(clusters)
final={}
no=0;

