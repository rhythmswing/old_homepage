---
layout: post
title:  "Pointwise Mutual Information and Word Embedding, and Graph Representation Learning"
date:   2018-04-16 07:29:54 +0800
categories: representation learning
---

# Introduction

One important question in machine learning is to represent effectively complex objects in computationally tractable approach. Common procedures are to assign these objects some vectors in a finite dimensional vector space, taking advantage of the geometric structure of the vector space. 


​	Examples include: words, sentences, documents in NLP; vertices in graphs; images; etc... In machine learning, people call it "embedding": word embedding, graph embedding. We call such assignment of vectors is called "embedding". Formally, given a set of objects \\(\mathcal{O}\\), the embedding is a mapping \\(\Phi:\mathcal{O}\rightarrow \mathcal{R}^d\\).

​	This is a summary of two representation algorithms: word2vec, which learns word embedding, and Deep2walk, which learns embedding for vertices in a social network. 

# Word2vec and Pointwise Mutual Information

​	For details of word2vec, please refer to [this tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). We analyze the skip-gram model which is recommended by Mikolov et al., the authors of the original paper of word2vec.

​	Suppose the vocabulary is \\(V\\). Each word in $V$ has a dual identity: it can be a word per se, or the *context* of some other word. We denote the set of words by \\(V_W\\) and context by \\(V_C\\). A popular approach of sampling word-context pairs is to define the context of $w$ as words which appear surrounding \\(w\\); we denote the collection of words and context pairs as \\(D\\). 

​	Word2vec gives each word two vector representations: one is for the word per se, and the other for the word as a context. That is to say, for each word $w\in V_W$, its representation is \\(\mathbf{w}\in\mathcal{R}^d\\), and for \\(c\in V_C\\), we have \\(\mathbf{c}\in\mathbf{R}^d\\). 

​	To learn the embedding, word2vec parameterized the occurrence of a word-context pair: $$ P((w,c)\in D) = \sigma(\mathbf{w}^T\mathbf{c}) $$

