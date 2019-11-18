from nltk.corpus import wordnet as wn


# print wn.synsets('cur')
# print wn.synsets('cut')
# print wn.synsets('knife')
# print wn.synsets('novel')
# print wn.synsets('book')
# print wn.synset('cut.v.01').lowest_common_hypernyms(wn.synset('knife.n.01'))
# l=wn.synsets('exam')
# print l
# for i in l:
# 	P=i.hypernyms()
# 	print P
# 	for p in P:
# 		O=p.hypernyms()
# 		print O
# 		for o in O:
# 			m=o.hypernyms()
# 			print m
# 			for q in m:
# 				p=q.hypernyms()
# 				print p
# 				for q in m:
# 					p=q.hypernyms()
# 					print p

# term1='student'
# term2='exam'
# def print_hypernyms(t,l):
# 	P=t.hypernyms()
# 	for p in P:
# 		print p,l
# 		print_hypernyms(p,l+1)


# l=wn.synsets(term1)
# for i in l:
# 	print_hypernyms(i,1)

# print "========="
# l=wn.synsets(term2)
# for i in l:
# 	print_hypernyms(i,1)



# find_level()



# print "_________"
# l=wn.synsets('student')
# print l
# for i in l:
# 	P=i.hypernyms()
# 	print P
# 	for p in P:
# 		print p.hypernyms()

# .lowest_common_hypernyms(wn.synset('student.n.01'))

# print wn.synset('exam.n.01').lowest_common_hypernyms(wn.synset('student.n.01'))



from nltk.corpus import wordnet
# import itertools as IT
# list1 = ["apple", "honey"]
# list2 = ["pear", "shell", "movie", "fire", "tree", "candle"]
# def f(word1, word2):
#     max_value=0
#     wordFromList1 = wordnet.synsets(word1)
#     wordFromList2 = wordnet.synsets(word2)
#     for wl1 in wordFromList1:
#     	for wl2 in wordFromList2:
#     		s = wl1.wup_similarity(wl2)
#     		if s > max_value: max_value = s 
#     print max_value
    # return(w1.name, wordFromList2.name, wordFromList1.wup_similarity(wordFromList2))

# for word1 in list1:
#     similarities=(f(word1,word2) for word2 in list2)
#     print similarities
#     print(max(similarities, key=lambda x: x[2]))
# print similarities    


def sim_f(word1, word2):
	if word1==word2:
		return 0
	max_value=0.0

	wordFromList1 = wordnet.synsets(word1)
	wordFromList2 = wordnet.synsets(word2)
	i=0
	for wl1 in wordFromList1:
		for wl2 in wordFromList2:
			print wl1,wl2,i
			p=wl1.wup_similarity(wl2)
    		print p
    		if max_value < p*1: 
				max_value = p
		i=i+1
	return max_value

print sim_f('google','google')	
