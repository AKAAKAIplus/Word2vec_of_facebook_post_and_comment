# -*- coding: utf-8 -*-
import pandas as pd
import os
from gensim.models import word2vec
import logging
import jieba


with open("filterwords.txt") as f:
	filter_word_content = f.readlines()
filter_word_content = [x.split('\n')[0] for x in filter_word_content] 


def main():
	output = open('seg.txt','w')

	count = 0
	for file in os.listdir("../data_pre_processing"):
		if str(file) != '.DS_Store':
			print(file)

			df = pd.read_csv(open( "../data_pre_processing/"+str(file),'rU' ) )
			df=df.values

			for i in df:
				if isinstance(i[0],str) and isinstance(i[1],str) and len(i[0])>0 and len(i[1])>0:
					i[0] = jieba.cut( i[0].strip(), cut_all=False)
					i[0] = "".join([word for word in list(i[0]) if word.encode('utf-8') not in filter_word_content])
					i[1] = jieba.cut( i[1].strip(), cut_all=False)
					i[1] = "".join([word for word in list(i[1]) if word.encode('utf-8') not in filter_word_content])


					for word in list(i[0]):
						output.write(word.encode('utf-8') +' ')
					output.write('\n')
					for word in i[1]:
						output.write(word.encode('utf-8') +' ')
					output.write('\n')
					if count % 2000 == 0:
						print("已完成 %d 筆資料" % count)

					count += 1

	output.close()
	
	sentences = word2vec.Text8Corpus("seg.txt")
	model = word2vec.Word2Vec(sentences, size=400)
	model.save("word2vector_400.model.bin")

if __name__ == "__main__":
	main()