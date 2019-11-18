WordCount={}
upper_words=["swapnil","asad","swapnil"]
for word in upper_words:
	if word in WordCount:
		WordCount[word]=WordCount[word]+1
		print WordCount,word,1
	else:
		WordCount[word]=1
    	print WordCount,word,2