---
layout: post
title:  "Graph Representation Learning"
date:   2018-05-04 07:29:54 +0800
categories: representation learning
---

**note: this article is still in progress.**

# Introduction
One important question in machine learning is to represent effectively complex objects in computationally tractable approach. Common procedures are to assign these objects some vectors in a finite dimensional vector space, taking advantage of the geometric structure of the vector space. 


Examples include: words, sentences, documents in NLP; vertices in graphs; images; etc... In machine learning, people call it "embedding": word embedding, graph embedding. We call such assignment of vectors is called "embedding". Formally, given a set of objects \\(\mathcal{O}\\), the embedding is a mapping \\(\Phi:\mathcal{O}\rightarrow \mathcal{R}^d\\). An embedding should be a good intermediary for some subsequent machine learning tasks. The term "embedding" should not be confused with the same term in graph theory; in fact, the definition of "embedding" in machine learning context seems to me a little bit casual. 

Representation learning is to "learn" vector representations for objects in the feature space, instead of handcrafting them. The idea is that it should be able to extract more flexible and general-purposed features for multiple tasks. Therefore, it is often desirable that representation learning rely on some intrinsic properties or similarity measure between objects, such as, in the case of networks, proximities. We then elaborate on usual proximities on graphs.

# What is a Graph?
A graph is a data structure usually defined as a tuple of sets: $G=(V,E)$ where $V$ is the set of vertices, and $E$ is the set of edges. When we consider graphs with finite set of vertices, which is the usual case, we would write $V=\\{v_1,\ldots,v_n\\}$ and $e_{ij}\in E$ as the edge between $v_i, v_j$, if indeed there is. 

Edges can be weighted. In usual settings, we use $w_{ij}\in \mathcal{R}^+$  to represent the weight of the edge between vertex $i$ and $j$. Even when edges are unweighted, this notion can also be utilized: in this case, it is simply a binary variable where $w_{ij}$ is there is an edge and $w_{ij}=0$ if there isn't. From $w_{ij}$'s we can construct a matrix $\mathbf{W}=(w_{ij})$, called the **adjacency matrix**.

In the above definition, we assumes that all vertices are essentially "of the same type". For example, in a social network, vertices are people, and edges represent whether people know each other. In a citation network, vertices could be authors, and egdes represent citation relationships. Such networks are **homogeneous**. There are also **heterogeneous** networks, where vertices are of different types. For example, a citation network's vertices could also consist of both authors and papers, instead of authors alone. In this case, edges represent different relationships: authors write paper, and papers cite one another. 

Graphs can also be attributed with additional information specific to each vertex, which is not representable by edges, such as age and gender of individuals in a social network. 

# Similarity on Graphs

How are two vertices in a graph similar to each other? The first immediate thought would be that two vertices are close if they are connected to each other. Proximity in this case is referred in literatures often as **first-order proximity**. 

Two vertices, when not connected, could also be similar, if they share a very similar neighborhood. In other words, vertices who are connected to the same other vertices are also close. This is referred to as  **second-order proximity**. 

There is no end to defining a larger scope of proximity. We can extend the idea to arbitrary $k$th-order proximity, by considering the set of $k$-hop paths from one vertex to antoher. 

The above notions of proximity are essentially based on the *local connectivity* of a vertex. Specifically, the whole idea of $k$th-order proximity is based on one vertex's neighborhood. As $k$ grows, the $k$-order neighborhood becomes more and more "global" in some sense. 

Nevertheless, the idea of proximity centered at the specified vertex, and there might be cases where the idea of proximity is not working so well. For example, two vertices could be similar to each other if they play the same role in the graph; in this case, they would share some **structural equivalence**. This would happen even if two vertices are in two separate components, where there is no path between the two. 

I would personally believe that preserving both local proximity and structural equivalence is, in general, not practical. There would be a trade-off if you want to jointly encode the two similarity measures in the same embedding. 


# Overview of Algorithms

Most graph embedding algorithms are based on one or more of the three methods to extract useful information:
1. Graph factorization techniques
2. Random walks on graphs
3. Deep learning

However, here we are not going to organize the articles according to these three categories. We will be introduced to the functionality of each of them as we proceed to meet more algorithms. 

## Methods on 1st Order Proximity
The most traditional methods utilizes 1sr order proximity of vertices in a network. Many of these are based on graph factorizations and are derived from geometric intuitions, which make them mathematically elegant (more or less). 

Examples are: Laplacian Eigenmap, Locally Linear Embedding, Graph Factorization, etc. Many these methods are based on **manifold learning** and non-linear dimensionality reduction. We take Laplacian Eigenmap as an example.

The core of laplacian eigenmap is quite simple. As a non-linear dimensinoality reduction algorithm, it involves constructing a graph from the original data points, and embedding the graph into a lower dimensional Euclidean space. Since we are only concerned with graph embedding, we omit the first step.

Say that the desired embedding in k-dimensional Euclidean space is $\mathbf{u}_i\in\mathcal{R}^k$ for vertex $v_i$. The objective of Laplacian Eigenmap is 

$$ \sum_{ij} w_{ij} \Vert u_i-u_j \Vert^2 $$

which, as we see, is an embodiment of preserving first-order proximity, i.e. the immediate neighborhood of every vertex.

Introducing some new matrices: $\mathbf{U} = [\mathbf{u}_1^T,\ldots,\mathbf{u}_n^T]$, and $\mathbf{D}$ who is a diagonal weight matrix whose entries are column-wise sums of $\mathbf{W}$. At last, the graph Laplacian is $\mathbf{L}\triangleq \mathbf{D}-\mathbf{W}$. 

By some calculation, we see that $\sum_{ij}w_{ij}\Vert u_i-u_j \Vert^2 = tr(\mathbf{U}^T\mathbf{L}\mathbf{U})$.

There is a trivial solution to the objective by making all $\mathbf{u}_i$ equal. This is obviously not what we want. Even if we demand that the vectors do not conincide, the solution is not unique, since we can always scale the vectors to make the objective smaller. Therefore we add the constraint: 
$$ \mathbf{U}^T \mathbf{D} \mathbf{U} = \mathbf{I} $$

The solution of this problem is given by finding the smallest $k$ non-zero eigenvalues and their corresponding eigenvectors. For example, say, these eigenvectors are $\mathbf{y}_1,\ldots,\mathbf{y}_k$, corresponding to eigenvalues $0<\alpha_1<\ldots<\alpha_k$. These eigenvectors spans a $k$ dimensional subspace. The optimal $\mathbf{U}$ is then $\mathbf{U}=[\mathbf{y}_1,\ldots,\mathbf{y}_k]$.

It is clear that Laplacian Eigenmap make uses of first-order proximity to obtain embedding which minimizes the weighted sum of vectors. 
Since real-world networks are often large-scale and sparse, tools for sparse matrix decomposition can be used to speed up efficiency.
However, when the network is really large, like, millions and billions of vertices, matrix decomposition becomes both time and memory inefficient even when sparsity is utilized. To put it shortly, Laplacian Eigenmap is not scalable. So are many other matrix factorization based methods. 

## 2nd Proximity: Connections to NLP

Context, random walk, and second order proximity.


# Information Perspective 
## Word2vec and Pointwise Mutual Information

​	For details of word2vec, please refer to [this tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). We analyze the skip-gram model which is recommended by Mikolov et al., the authors of the original paper of word2vec.

​	Suppose the vocabulary is \\(V\\). Each word in \\(V\\) has a dual identity: it can be a word per se, or the *context* of some other word. We denote the set of words by \\(V_W\\) and context by \\(V_C\\). A popular approach of sampling word-context pairs is to define the context of \\(w\\) as words which appear surrounding \\(w\\); we denote the collection of words and context pairs as \\(D\\). 

​	Word2vec gives each word two vector representations: one is for the word per se, and the other for the word as a context. That is to say, for each word \\(w\in V_W\\), its representation is \\(\mathbf{w}\in\mathcal{R}^d\\), and for \\(c\in V_C\\), we have \\(\mathbf{c}\in\mathbf{R}^d\\). 

​	To learn the embedding, word2vec parameterized the occurrence of a word-context pair: 

$$ P((w,c)\in D) = \sigma(\mathbf{w}^T\mathbf{c}) $$

# Semi-supervised learning 

# Representation with Attributes