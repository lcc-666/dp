import gensim

sentences = [['this', 'is', 'a', 'hot', 'pie'], ['this', 'is', 'a', 'cool', 'pie'], ['this', 'is', 'a', 'red', 'pie'],
             ['this', 'is', 'not', 'a', 'cool', 'pie']]
model = gensim.models.Word2Vec(sentences, min_count=1)

print(model.wv['this'])
print(model.wv['is'])
print('vector size:', len(model.wv['is']))
print(model.wv.similarity('this', 'is'))
print(model.wv.similarity('this', 'not'))
print(model.wv.similarity('this', 'a'))
print(model.wv.similarity('this', 'hot'))
print(model.wv.similarity('this', 'cool'))
print(model.wv.similarity('this', 'pie'))

print(model.wv.most_similar(positive=['cool', 'red'], negative=['this']))
