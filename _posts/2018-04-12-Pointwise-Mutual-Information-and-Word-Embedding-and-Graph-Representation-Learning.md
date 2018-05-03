---
layout: post
title:  "Graph Representation Learning"
date:   2018-04-16 07:29:54 +0800
categories: representation learning
---

**note: this article is still in progress.**
# Introduction
One important question in machine learning is to represent effectively complex objects in computationally tractable approach. Common procedures are to assign these objects some vectors in a finite dimensional vector space, taking advantage of the geometric structure of the vector space. 


Examples include: words, sentences, documents in NLP; vertices in graphs; images; etc... In machine learning, people call it "embedding": word embedding, graph embedding. We call such assignment of vectors is called "embedding". Formally, given a set of objects \\(\mathcal{O}\\), the embedding is a mapping \\(\Phi:\mathcal{O}\rightarrow \mathcal{R}^d\\). An embedding should be a good intermediary for some subsequent machine learning tasks. The term "embedding" should not be confused with the same term in graph theory; in fact, the definition of "embedding" in machine learning context seems to me a little bit casual. 

Representation learning is to "learn" vector representations for objects in the feature space, instead of handcrafting them. The idea is that it should be able to extract more flexible and general-purposed features for multiple tasks. Therefore, it is often desirable that representation learning rely on some intrinsic properties or similarity measure between objects, such as, in the case of networks, proximities. We then elaborate on usual proximities on graphs.

# What is a Graph?
A graph is a data structure usually defined as a tuple of sets: $G=(V,E)$ where $V$ is the set of vertices, and $E$ is the set of edges. When we consider graphs with finite set of vertices, which is the usual case, we would write $V=\\{v_1,\ldots,v_n\\}$ and $e_{ij}\in E$ as the edge between $v_i, v_j$, if indeed there is. 

In the above definition, we assumes that all vertices are essentially "of the same type". For example, in a social network, vertices are people, and edges represent whether people know each other. In a citation network, vertices could be authors, and egdes represent citation relationships. Such networks are **homogeneous**. There are also **heterogeneous** networks, where vertices are of different types. For example, a citation network's vertices could also consist of both authors and papers, instead of authors alone. In this case, edges represent different relationships: authors write paper, and papers cite one another. 

Graphs can also be attributed with additional information specific to each vertex, which is not representable by edges, such as age and gender of individuals in a social network. 

# Similarity on Graphs

How are two vertices in a graph similar to each other? The first immediate thought would be that two vertices are close if they are connected to each other. Proximity in this case is referred in literatures often as **first-order proximity**. 

Two vertices, when not connected, could also be similar, if they share a very similar neighborhood. In other words, vertices who are connected to the same other vertices are also close. This is referred to as  **second-order proximity**. 

There is no end to defining a larger scope of proximity. We can extend the idea to arbitrary $k$th-order proximity, by considering the set of $k$-hop paths from one vertex to antoher. 

The above notions of proximity are essentially based on the *local connectivity* of a vertex. Specifically, the whole idea of $k$th-order proximity is based on one vertex's neighborhood. As $k$ grows, the $k$-order neighborhood becomes more and more "global" in some sense. 

Nevertheless, the idea of proximity centered at the specified vertex, and there might be cases where the idea of proximity is not working so well. For example, two vertices could be similar to each other if they play the same role in the graph; in this case, they would share some **structural equivalence**. This would happen even if two vertices are in two separate components, where there is no path between the two. 

I would personally believe that preserving both local proximity and structural equivalence is, in general, not practical. There would be a trade-off if you want to jointly encode the two similarity measures in the same embedding. 

# Connections to NLP

# Some Algorithms
## Laplacian Eigenmap
## Graph Factorization
## Deepwalk
## LINE

## SDNE

# Information Perspective 
## Word2vec and Pointwise Mutual Information

​	For details of word2vec, please refer to [this tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). We analyze the skip-gram model which is recommended by Mikolov et al., the authors of the original paper of word2vec.

​	Suppose the vocabulary is \\(V\\). Each word in \\(V\\) has a dual identity: it can be a word per se, or the *context* of some other word. We denote the set of words by \\(V_W\\) and context by \\(V_C\\). A popular approach of sampling word-context pairs is to define the context of \\(w\\) as words which appear surrounding \\(w\\); we denote the collection of words and context pairs as \\(D\\). 

​	Word2vec gives each word two vector representations: one is for the word per se, and the other for the word as a context. That is to say, for each word \\(w\in V_W\\), its representation is \\(\mathbf{w}\in\mathcal{R}^d\\), and for \\(c\in V_C\\), we have \\(\mathbf{c}\in\mathbf{R}^d\\). 

​	To learn the embedding, word2vec parameterized the occurrence of a word-context pair: 

$$ P((w,c)\in D) = \sigma(\mathbf{w}^T\mathbf{c}) $$

# Semi-supervised learning 

# Representation with Attributes