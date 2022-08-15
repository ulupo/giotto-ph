---
title: '*giotto-ph*: A Python Library for High-Performance Computation of 
Persistent Homology of Vietoris–Rips Filtrations'
tags:
  - Python
  - topological data analysis
  - persistent homology
  - machine learning
  - Vietoris–Rips filtrations
  - concurrency
  - simplicial collapses
authors:
  - name: Julián Burella Pérez
    orcid: 0000-0003-0631-1179
    equal-contrib: true
    affiliation: 1
  - name: Sydney Hauke
    orcid: 0000-0003-3810-5652
    equal-contrib: true
    affiliation: 1
  - name: Umberto Lupo
    orcid: 0000-0001-6767-493X
    equal-contrib: true
    corresponding: true
    affiliation: "2, 3"
  - name: Matteo Caorsi
    orcid: 0000-0001-9416-9090
    affiliation: 4
  - name: Alberto Dassatti
    orcid: 0000-0002-5342-3723
    affiliation: 1
affiliations:
  - name: HEIG-VD, HES-SO, Route de Cheseaux 1, Yverdon-les-Bains, Switzerland
    index: 1
  - name: Institute of Bioengineering, School of Life Sciences, École Polytechnique Fédérale de Lausanne (EPFL), CH-1015 Lausanne, Switzerland
    index: 2
  - name: SIB Swiss Institute of Bioinformatics, CH-1015 Lausanne, Switzerland
    index: 3
  - name: L2F SA, Rue du centre 9, Saint-Sulpice, Switzerland
    index: 4
date: 24 July 2022
bibliography: paper.bib
---

# Summary

We introduce *giotto-ph*, a high-performance package for the computation of 
Vietoris–Rips (VR) persistent homology barcodes. *giotto-ph*'s C++ backend 
builds on a recent [@morozov2020towards; @morozov2020lock] lockfree (multicore) 
implementation of *Ripser* [@bauer2021ripser], while borrowing ideas from 
*Ripser++* [@zhang2020gpuaccelerated] to increase parallelization opportunities. 
In addition, we introduce novel algorithmic speedups and implement the retrieval 
of flag persistence generators, thus paving the way for high-performance 
*differentiable* VR barcode computations. Furthermore, we improve portability, 
provide convenient Python bindings inspired by *Ripser.py* [@ctralie2018ripser], 
integrate a reimplementation of the *GUDHI* library's *Edge Collapser* 
[@boissonnat2020edge; gudhi:Collapse] as a pre-processing step, and provide 
support for weighted VR filtrations [@anai2020dtmbased]. Our persistent homology 
backend establishes a new state of the art, surpassing even GPU-accelerated 
implementations such as *Ripser++* when using as few as 5–10 CPU cores.

# Statement of need \label{sec:need}

*Persistent homology* (PH) (see e.g. [@ghrist2007barcodes; 
@edelsbrunner2008persistent; @edelsbrunner2014persistent; 
@oudot2015persistence; @chazal2016structure; @perea2018brief; 
@carlsson2019persistent; @nanda2021computational] for surveys) has led to the 
adoption of novel topological methodologies in a wide variety of computational 
contexts, such as geometric inference 
[@edelsbrunner2014short; @boissonnat2018geometric], signal processing 
[@robinson2014topological; @perea2015sliding], data visualization 
[@tierny2018topological], data analysis and machine learning 
[@carlsson2009topology; @chazal2021introduction; @hensel2021survey]. Among the 
invariants described by this theory, the (*persistence*) *barcode* 
[@frosini1990shapes; @frosini1992measuring; @barannikov1994morse; 
@robins1999approximations; @edelsbrunner2000simplification; 
@zomorodian2005computing] has attracted the most attention due to (a) its 
ability to track the appearance and disappearance of holes, voids, or 
higher-dimensional topological features in data, (b) its succinct nature and 
ease of representation, (c) its robustness under perturbations 
[@damico2003optimal; @cohen-steiner2007stability], and (d) its amenability to 
algorithmic optimization – see Sec. 1 in [@bauer2021ripser] for a review, and 
the recent implementations in [@aggarwal2021dory; @vonbromssen2021computing].

Despite these successes, the computation of barcodes remains a challenge when
dealing with large datasets and/or with high-dimensional topological features. 
Indeed, the input to any barcode computation is a growing, one-parameter family 
of combinatorial objects, called a *filtration*, and several filtrations of 
interest in applications become very large as their defining parameter 
increases. This leads to a staggering number of operations required to distil 
barcodes via established matrix reduction algorithms.

This is particularly true of *Vietoris–Rips* (VR) filtrations of finite metric 
spaces and of *flag* filtrations of weighted undirected graphs, which are the 
object of this work.

# Related work \label{sec:related_work}

To the best of our knowledge, at the time of writing 
[*Ripser*](https://github.com/Ripser/ripser) [@bauer2021ripser] is the *de 
facto* state of the art and reference for computing VR persistence barcodes 
on CPUs. *Ripser* uses multiple known optimizations like *clearing* 
[@chen2011persistent] and *cohomology* [@desilva2011dualities]. Furthermore, 
it makes use of other performance-oriented ideas, such as the implicit 
representation of the (co)boundary and reduced (co)boundary matrices, and 
the *emergent/apparent pairs* optimizations (we refer to [@bauer2021ripser] 
for definitions and details). At the time of writing, the latest version of 
*Ripser* is *v1.2.1* (release date: 22 May 2021).

Although *Ripser v1.2.1* is arguably the fastest existing code for computing VR
barcodes in a sequential (i.e., single CPU core) setting, it has no parallel 
capabilities. Overcoming this limitation is possible as two recent lines of 
work [@morozov2020towards; @zhang2020gpuaccelerated] demonstrate. Based on a 
"pairing uniqueness lemma" proved in [@cohen-steiner2006vines], Morozov and 
Nigmetov [@morozov2020towards] observe that the reduction of the (co)boundary 
matrix can, in fact, be performed out of order as long as column additions are 
always performed left to right (relative to the filtration order). Therefore – 
these authors suggest – any column reduction can be efficiently performed in \
parallel provided adequate synchronisation is used. A functional proof of 
concept of this idea as applied to a now superseded (in particular, based on 
*Ripser v1.1* and hence not using apparent pairs, cf. \autoref{fig:lib}) 
version of *Ripser* was put in the public domain in June 2020 
[@morozov2020lock], but has not led to a distributable software package. A 
generic implementation of the ideas in [@morozov2020towards], not tied to 
Vietoris–Rips filtrations and instead designed to make lock-free reduction 
possible on any (co)boundary matrix, has recently been published as the 
*Oineus* library [@nigmetov2020oineus].

*Ripser++* [@zhang2020gpuaccelerated] implements the idea of finding apparent 
pairs in parallel on a GPU to accelerate the computation of VR barcodes. But,
the reduction matrix step is executed sequentiall on the CPU.

All implementations presented in this subsection so far (barring 
[@nigmetov2020oineus], which also provides some Python bindings) are written in 
low-level languages (C++, CUDA). *Ripser.py* [@ctralie2018ripser] contains a 
modified version of *Ripser* with support for non-zero birth times and for the 
retrieval of cocycles, as well as a convenient Python interface.

Meanwhile, in 2020, Boissonnat and Pritam presented a new algorithm they called 
*Edge Collapser* (EC) [@boissonnat2020edge]. Independently of the code used to 
compute barcodes, EC can be used as a pre-processing step on any flag filtration 
to remove "redundant" edges – and modify the filtration values of some others
– while ensuring that the flag filtration obtained from the thus "sparsified"
weighted graph has the same barcode as the original filtration. Although it
introduces an initial overhead, pre-processing by EC can dramatically improve 
the end-to-end run-time for barcode computation by greatly reducing the
complexity of the downstream reduction steps. As reported by those authors, this
is especially true when one wishes to compute barcodes in high homology
dimensions, and/or when one is dealing with large datasets. An implementation of 
EC has already been integrated into the *GUDHI* library 
[@gudhi:urm; @gudhi:Collapse].

Gudhi improved their first version of EC implementation with new techniques.
They compared [@glisse2022edge] their new version against our implementation.
The achieved outstading speed-ups, we hope at some point integrate their latest version.

# Our contribution

In this context, we present 
[*giotto-ph*](https://github.com/giotto-ai/giotto-ph), a Python package built 
on top of a C++ backend that computes PH barcodes for VR filtrations on the CPU. 
To the best of our knowledge, this is the first package fully integrating the 
three ideas described in 
\hyperref[sec:related_work]{``Related work"\ref*{sec:related_work}} (lock-free 
reduction, parallelized search for apparent pairs, edge collapses) in a single 
portable, easy-to-use library. We remark that, after the release of our code 
and of the first version of this paper, we learned about a very recent thesis 
[@tulchinskii2021fast] and associated 
[publicly available code](https://github.com/ArGintum/PersistenceHomology)
(retrieved 1 August 2021) in which a very similar program to ours has been 
carried out, though the apparent pairs optimization and support for coefficients 
in finite prime fields other than $\mathbb{F}_2$ are among the features still 
missing from that package.

When developing *giotto-ph* we focused on increasing execution speed throughout. 
In particular:

  1. We built on the ideas for parallel reduction presented in 
     [@morozov2020towards] and on the prototype implementation described in 
     [@morozov2020lock], and improved execution speed and resource usage by 
     implementing custom lock-free hash tables and a thread pool.
  2. Similarly to *Ripser++*, we implemented a parallel version of the 
     apparent pairs optimization, thus far only present in serial form in 
     *Ripser v1.2*.
  3. We re-implemented the EC algorithm to increase its execution speed 
     compared to [@gudhi:Collapse]. The simple observation that the 
     well-known *enclosing radius* optimization is applicable to EC is shown 
     here to lead to even larger improvements.

Our results show that our code is often $1.5$ to $2$ times (and, in one example, 
almost $8$ times) faster than [@morozov2020towards] and able to beat 
*Ripser++* [@zhang2020gpuaccelerated], the current state-of-the-art GPU 
implementation, while running only on CPU and with as few as 5–10 cores.

Finally, *giotto-ph* owes some architectural decisions to *Ripser.py* 
[@ctralie2018ripser] – in particular, the support for node weights. At the 
level of the Python interface, our main contribution is supporting *weighted 
Rips filtrations* – in particular, the *distance-to-measure*–based filtrations 
described in [@anai2020dtmbased].

Thanks to its reduced memory usage and shorter run-times, we hope that 
*giotto-ph* will enable researchers to explore larger datasets, and in higher 
homology dimensions, than ever before.

# Implementation

![*giotto-ph* consists of a C++ backend and a Python frontend. 
The Python interface is based on *Ripser.py* [@ctralie2018ripser] (see 
\hyperref[sec:python]{``Python interface"\ref*{sec:python}} for details). The 
figure also shows the inheritance of *giotto-ph*'s C++ backend from pre-dating 
implementations. \label{fig:lib}](architecture_bpj.svg){width=100%}

Our aim with *giotto-ph* is to provide an alternative to the excellent *Ripser.py* library,
retaining several of the latter's advantages, namely portability and ease of
use, while replacing the C++ backend with a new parallel and higher-performance
version.

## C++ backend \label{sec:Cpp_backend}

The implementation of *giotto-ph*'s backend parallel algorithm is heavily 
inspired by [@morozov2020lock], a functional proof of concept of 
[@morozov2020towards]. Starting from [@morozov2020towards], we replaced the 
main data structure and the threading strategy to minimize the overhead 
introduced by adding parallelism. Furthermore, we introduced the apparent 
pairs approach, in its parallel form, to harvest its benefits in shortening 
run-times: a decreased number of columns to reduce [@bauer2021ripser] and an 
additional early stop condition when enumerating cofacets.

The final component in our C++ backend is a rewriting of the EC algorithm,
implemented so far only in the *GUDHI* library [@gudhi:Collapse]. Our 
implementation focuses on performance and adds support to
weighted graphs with arbitrary (possibly non-positive) edge weights as well as 
arbitrary node weights.

## Python interface \label{sec:python}

Our Python interface is based on *Ripser.py* [@ctralie2018ripser]. While it 
lacks some of *Ripser.py*'s features, such as the support for "greedy 
permutations", it introduces the following improvements:

  - Support for Edge Collapser. EC is disabled by default, but users can easily 
    enable it by means of the `collapse_edges` optional argument.

  - Support for enclosing radius. The *(minimum) enclosing radius* of a 
    finite metric space is the radius of the smallest enclosing ball of that 
    space. Simplices with higher filtration values than the enclosing radius 
    can be safely omitted without changing the final barcode.

  - Weighted VR filtrations. Distance-to-measure (DTM) based filtrations 
    [@anai2020dtmbased] address re-weight vertices and distances according to 
    the local neighbourhood structure, to yield a barcode which is more robust 
    to statistical outliers. The user can toggle DTM-based reweighting (or 
    more general reweightings) by setting the optional parameters `weights` and
  `weight_params`.

# Experimental results \label{sec:experiments}

All experiments presented in this paper were performed on a machine running 
Linux CentOS 7.9.2009 with kernel 5.4.92, equipped with two Intel® XEON® 
Gold 6248R (24 physical cores each) and a total of 128 GB of RAM. 

We present measures on the datasets of \autoref{tbl:datasets} because they 
are publicly available, and they are used in publications [@Otter_2017; 
@bauer2021ripser] describing established algorithms, making them a 
representative benchmark set and facilitating comparisons among competing 
solutions. All datasets are stored as point clouds. When the `threshold` 
parameter is empty, the tests report run-times with the enclosing radius 
option active. The `dim` parameter corresponds to the maximum dimension for 
which we compute PH, and the `coeff` parameter corresponds to the prime 
field of coefficients (in our tests, this is always $\mathbb{F}_2$).

\begin{table}
\centering
\caption{Datasets used for benchmarking. ``Size" means the number of points in 
the dataset.}
\label{tbl:datasets}
\begin{tabular}{lrrrr}
    \hline
    \textbf{dataset} & \textbf{size} & \textbf{threshold} & \textbf{dim} & \textbf{coeff} \\ \hline
    \texttt{sphere3}  & 192   &      & 2 & 2 \\ \hline
    \texttt{dragon}   & 2000  &      & 1 & 2 \\ \hline
    \texttt{o3\_1024} & 1024  & 1.8  & 3 & 2 \\ \hline
    \texttt{random16} & 50    &      & 7 & 2 \\ \hline
    \texttt{fractal}  & 512   &      & 2 & 2 \\ \hline
    \texttt{o3\_4096} & 4096  & 1.4  & 3 & 2 \\ \hline
    \texttt{torus4}   & 50000 & 0.15 & 2 & 2 \\
    \hline
\end{tabular}
\end{table}

## Comparison with state-of-the-art algorithms

In this section we compare our implementation with other implementations of 
PH for VR filtrations that use an approach similar to Ripser. We do not 
directly compare with other existing libraries which adopt different 
approaches, like *GUDHI* [@gudhi:urm] and *Eirene* [@henselmanghristl6], 
because from [@bauer2021ripser] it is evident that *Ripser* is always faster. 

\autoref{fig:gph_vs_ripser_1.2} compares the *giotto-ph* backend and 
*Ripser v1.2*. When the computation of the filtration is very fast, due to the 
reduced number of points or the low dimension of the computation, there is 
marginal or no benefit in adopting our parallel approach.

![Speed-up of *giotto-ph* compared to *Ripser v1.2*. *giotto-ph* is always 
faster than *Ripser v1.2* when using more than one thread. For the `sphere3` 
dataset, there is little speed-up in general, and virtually no speed-up 
($\sim 1.05$) with 13 threads or above; in that case we compute homology 
only up to dimension $1$ and the cost of setting up the parallel element of 
the library is non-zero.
\label{fig:gph_vs_ripser_1.2}](giotto_vs_ripser1.2.svg){width=80%}

According to our measurements reported in Figure \autoref{fig:moro}, our 
implementation outperforms Morozov and Nigmetov's proof-of-concept 
implementation [@morozov2020lock] in most cases, and most noticeably when 
the number of parallel resources increases. The only exception when using 
multiple threads is `sphere3`. The version in [@morozov2020lock] performs 
better and better on `sphere3` when increasing the number of parallel 
resources, while ours does not. The main culprit is
that, while in [@morozov2020lock] parallel resources are allocated only when 
needed in the computation, our thread pool will allocate
all the parallel resources indicated by the user ahead of time. Our approach is 
most beneficial when the allocated resources can be reused during the 
computation, and this is true e.g. when computing homology dimensions in degree
$2$ and above. However, when computing only up to dimension $1$, it is only 
necessary to allocate the parallel resources once, and an on-the-fly approach 
such as the one in [@morozov2020lock] can be faster. Another logically 
independent reason for this observed performance loss has to do with apparent 
pairs: we remind the reader that the implementation in [@morozov2020lock] is 
based upon *Ripser v1.1* which, unlike *Ripser v1.2* considered here, did not 
make use of the apparent pairs optimization. While the search for apparent 
pairs and subsequent column assembly step is performed in parallel in homology 
dimension $1$ or higher, it is only done serially in dimension $0$.

![Speed-up of *giotto-ph* compared to the implementation in 
[@morozov2020lock]. *giotto-ph* is faster in general, but with the `fractal` 
dataset the speed-up is larger – almost a factor of $8$ when $48$ threads 
are used. This phenomenon is explained by the large number of apparent pairs 
in this specific dataset. On the other hand, performance on `sphere3` is 
worse as explained in the main body of text.
\label{fig:moro}](giotto_vs_lf.svg){width=80%}

Considering the good performance obtained, we decided to compare our 
implementation with the state-of-the-art parallel code running on GPU: 
*Ripser++* [@zhang2020gpuaccelerated]. For this test, we ran our code on the 
same datasets used in [@zhang2020gpuaccelerated] (for full details check 
Table 2 on page 23 of [@zhang2020gpuaccelerated]) and compared our run-times 
with the reported figures. \autoref{fig:comparison_gph_rpp} shows that on our 
test machine, we achieve better performance when using only 4 to 10 threads, 
depending on the dataset, confirming that a relatively new CPU with at least 8 
cores should be able to beat a high-end GPU on this computation. 

There are multiple limitations in *Ripser++* that were addressed in 
*giotto-ph*. First, *Ripser++* does not perform the matrix reduction in 
parallel. Second, apparent pairs are stored in a sorted array in order to 
provide apparent pair lookups in $\mathcal{O}(\log{}n)$ time using binary 
searches. Since it is possible to carry out the matrix reduction without 
recording and/or sorting apparent pairs, *giotto-ph* results in a 
competitive solution, even if running on less high-performance hardware.

![Run times comparison of *giotto-ph* (full blue line) and *Ripser++* 
(dashed orange line) using datasets from [@zhang2020gpuaccelerated]. The 
$x$-axis represent the number of threads used and the $y$-axis  the time (in 
seconds) to complete the PH computation.
\label{fig:comparison_gph_rpp}](giotto-ph_rpp.svg){width=100%}

# Acknowledgements

We thank Anibal Medina-Mardones and Kathryn Hess Bellwald for numerous fruitful 
discussions, as well as Ulrich Bauer for very helpful conversations about 
\textit{Ripser}. This work was supported by the Swiss Innovation Agency 
(Innosuisse project 41665.1 IP-ICT).

# References


