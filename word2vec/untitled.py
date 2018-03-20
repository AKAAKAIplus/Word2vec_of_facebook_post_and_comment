# -*- coding: utf-8 -*-
import pandas as pd
import os
from gensim.models import word2vec
import logging
import jieba

model = word2vec.Word2Vec.load("word2vector_400.model.bin")
print(len(model.wv['一'.decode('utf-8')]))