import numpy as np
import sklearn.cluster
import distance


f=open('data/testq.txt','r')
text=f.read();
words = text.split(" ")
inp_stop=open('data/stopwords.txt','r+') #read stopword from file
stopwords=eval(inp_stop.read())
ws=[]
for word in words:
	if word not in stopwords:
		ws.append(word)


# print ws
lev_similarity = np.array([[float(1)/(distance.levenshtein(w1,w2)+1) for w1 in words] for w2 in words])



