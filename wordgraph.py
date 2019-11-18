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

inp_stop=open('data/stopwords.txt','r+') #read stopword from file
stopwords=eval(inp_stop.read())
# print stopwords


#Intialization
id=-1
i=0
directory="data/text"
II={}
Document_TF_Count={}
Documents=os.listdir(directory)


for Document in Documents:
	if i<4:
		i=i+1
	else:
		break
	f=open(directory+"/"+Document,'r');
	# c=f.read()
	c=f.read().split('\n')
	f.close();
	# id=c[4].replace('DOC ID:','')
	id=id+1
	content=c[5].replace('CONTENT:','')
	#print c
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

#Assign Term Frequency as Zero
vocab=II.keys()
for v in vocab:
	for key in Document_TF_Count:
		if v not in Document_TF_Count[key]:
			Document_TF_Count[key][v]=0


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
# print Document_TF_Count
#Calculate TF-IDF
#print Document_TF_Count.keys()



with open('Document_TF_Count', 'wb') as fp:
    pickle.dump(Document_TF_Count, fp)

with open ('Document_TF_Count', 'rb') as fp:
    Document_TF_Count = pickle.load(fp)


term_vector={}
for v in vocab:
	idf=math.log((float(id+1)/len(II[v])),10)
	array=[]
	for key in Document_TF_Count:
		k=int(key)
		array+=[Document_TF_Count[key][v]*idf]
	term_vector[v]=array
#print "Term Vector without sumation similarity"
#pprint.pprint(term_vector)
#~ print "++++++++++++++++++"
#Update TF-IDF using TMI
with open('termvector', 'wb') as fp:
    pickle.dump(term_vector, fp)

with open ('termvector', 'rb') as fp:
    term_vector = pickle.load(fp)

def sim_f(word1, word2):
	if word1==word2:
		return 0
	max_value=0.0
	wordFromList1 = wordnet.synsets(word1)
	wordFromList2 = wordnet.synsets(word2)
	for wl1 in wordFromList1:
		#~ if wl1 in wordFromList2:
			#~ return 1
	#~ return 0	
		for wl2 in wordFromList2:
			p=wl1.wup_similarity(wl2)
			if max_value < p: 
				max_value = p
	return max_value

sim={}
for term1 in term_vector:
	sim[term1]={}
	for term2 in term_vector:
		sim[term1][term2]=sim_f(term1,term2)

with open('sim', 'wb') as fp:
    pickle.dump(sim, fp)

with open ('sim', 'rb') as fp:
    sim = pickle.load(fp)
#print "Similirity",sim
#print "++++++++++++++++++"

for term in term_vector:
 	l=len(term_vector[term])
 	#~ print term
 	for i in range(l):
		sumation=0
		for v in vocab:
			sumation+=term_vector[v][i]*sim[term][v]	
		term_vector[term][i]+=sumation		
#print "Term Vector sumation similarity"
#pprint.pprint(term_vector)

Matrix={}
for tv1 in term_vector:
	Matrix[tv1]={}
	for tv2 in term_vector:
		Matrix[tv1][tv2] =1-spatial.distance.cosine(term_vector[tv1],term_vector[tv2])



with open('matrix', 'wb') as fp:
    pickle.dump(Matrix, fp)

with open ('matrix', 'rb') as fp:
    Matrix = pickle.load(fp)

Distance_Matrix=np.array(Matrix)
#print "Distance Matrix"
#pprint.pprint(Distance_Matrix)

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
		DM[i][j]=Matrix[l1][l2]
		j=j+1
	i=i+1	
vocab=np.asarray(vocab)	

print DM
#~ for v in vocab:
	
#print "Labels",labels

#~ print DM	
	
#~ #MCL Start	
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

# import csv

# # write it
# with open('test_file.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     [writer.writerow(r) for r in DM]

# # read it
# with open('test_file.csv', 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     D = [[e for e in r] for r in reader]
# D=np.array(D)

# np.savetxt('test.out', DM, delimiter=',') 
# D=np.loadtxt('test.out',dtype=float)


M, clusters = mcl_c.mcl(DM, expand_factor = 5, inflate_factor = 2, max_loop = 10, mult_factor = 5)
print "Clusters",clusters
pos=mcl_c.draw(G,M,clusters,labels)
#~ #MCL END


#~ print "swapnil"
#~ print M
#~ print "========"
#~ print clusters                   
#~ affprop = sklearn.cluster.AffinityPropagation(preference=-50,affinity="precomputed", damping=0.5)
#~ affprop = sklearn.cluster.AffinityPropagation(preference=-565,affinity="euclidean",convergence_iter=5, damping=0.5)
#~ affprop.fit(DM)
#~ count=0
#~ for cluster_id in np.unique(affprop.labels_):
    #~ exemplar = vocab[affprop.cluster_centers_indices_[cluster_id]]
    #~ cluster = np.unique(vocab[np.nonzero(affprop.labels_==cluster_id)])
    #~ cluster_str = ", ".join(cluster)
    #~ print(cluster_str)
    #~ count=count+1
#~ print count

#~ import matplotlib.pyplot as plt

#~ from sklearn.feature_extraction import image
#~ from sklearn.cluster import spectral_clustering

#~ img=100*np.array(DM)
#~ print "++++++++++++++++++++++++",vocab,"++++++++++++++++++"
#~ # We use a mask that limits to the foreground: the problem that we are
#~ # interested in here is not separating the objects from the background,
#~ # but separating them one from the other.
#~ mask = img.astype(bool)

#~ img = img.astype(float)
#~ img += 1 + 0.2 * np.random.randn(*img.shape)

#~ # Convert the image into a graph with the value of the gradient on the
#~ # edges.
#~ graph = image.img_to_graph(img, mask=mask)
#~ print img
#~ print "======================="
#~ print graph
#~ # Take a decreasing function of the gradient: we take it weakly
#~ # dependent from the gradient the segmentation is close to a voronoi
#~ graph.data = np.exp(-graph.data / graph.data.std())

#~ # Force the solver to be arpack, since amg is numerically
#~ # unstable on this example
#~ labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
#~ label_im = -np.ones(mask.shape)
#~ label_im[mask] = labels

#~ plt.matshow(img)
#~ plt.matshow(label_im)

#~ plt.show()



#~ from time import time

#~ import numpy as np
#~ from scipy import ndimage
#~ from matplotlib import pyplot as plt

#~ from sklearn import manifold, datasets


#~ X_red=np.random.rand(32,32)
#~ #----------------------------------------------------------------------
#~ # Visualize the clustering
#~ def plot_clustering(X_red, X, labels, title=None):
    #~ x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    #~ X_red = (X_red - x_min) / (x_max - x_min)

    #~ plt.figure(figsize=(6, 4))
    #~ for i in range(X_red.shape[0]):
        #~ plt.text(X_red[i, 0], X_red[i, 1], str(i),
                 #~ color=plt.cm.spectral(labels[i] / 10.),
                 #~ fontdict={'weight': 'bold', 'size': 9})

    #~ plt.xticks([])
    #~ plt.yticks([])
    #~ if title is not None:
        #~ plt.title(title, size=17)
    #~ plt.axis('off')
    #~ plt.tight_layout()

#~ #----------------------------------------------------------------------

#~ from sklearn.cluster import AgglomerativeClustering

#~ for linkage in ('ward', 'average', 'complete'):
    #~ clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    #~ t0 = time()
    #~ clustering.fit(DM)
    #~ print("%s : %.2fs" % (linkage, time() - t0))

    #~ plot_clustering(DM, DM, clustering.labels_, "%s linkage" % linkage)


#~ plt.show()
