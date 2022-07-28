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

We introduce *giotto-ph*, a high-performance, open-source software package 
for the computation of Vietoris--Rips barcodes. *giotto-ph* is based on 
Morozov and Nigmetov's lockfree (multicore) implementation of Ulrich Bauer's 
*Ripser* package. It also contains a re-working of the *GUDHI* library's 
implementation of Boissonnat and Pritam's *Edge Collapser*, which can be 
used as a pre-processing step to dramatically reduce overall run-times in 
certain scenarios. Our contribution is twofold: on the one hand, we 
integrate existing state-of-the-art ideas coherently in a single library and 
provide Python bindings to the C++ code. On the other hand, we increase 
parallelization opportunities and improve overall performance by adopting 
more efficient data structures. Our persistent homology backend establishes 
a new state of the art, surpassing even GPU-accelerated implementations such 
as *Ripser++* when using as few as 5–10 CPU cores. Furthermore, our 
implementation of *Edge Collapser* has fewer software dependencies and 
improved run-times relative to *GUDHI*'s original implementation.

# Statement of need \label{sec:need}

In recent years, *persistent homology* (PH) (see e.g. [@ghrist2007barcodes; 
@edelsbrunner2008persistent; @edelsbrunner2014persistent; 
@oudot2015persistence; @chazal2016structure; @perea2018brief; 
@carlsson2019persistent; @nanda2021computational] for surveys) has been a key 
driving force behind the ever-increasing adoption of topological approaches in 
a wide variety of computational contexts, such as geometric inference 
[@edelsbrunner2014short; @boissonnat2018geometric], signal processing 
[@robinson2014topological; @perea2015sliding], data visualization 
[@tierny2018topological], and, more generally, data analysis 
[@carlsson2009topology; @chazal2021introduction] and machine learning 
[@hensel2021survey]. Among the main invariants described by this theory, the 
(*persistence*) *barcode* [@frosini1990shapes; @frosini1992measuring; 
@barannikov1994morse; @robins1999approximations; 
@edelsbrunner2000simplification; @zomorodian2005computing] has attracted the 
most attention due to (a) its ability to track the appearance and 
disappearance of holes, voids, or higher-dimensional topological features in 
data, throughout entire ranges of parameters, (b) its succinct nature and ease 
of representation, as it simply consists of a (typically small) collection of 
intervals of the real line, (c) its provable robustness under perturbations of 
the input data [@damico2003optimal; @cohen-steiner2007stability], and (d) its 
amenability to computation and algorithmic optimization, as demonstrated by the 
large number of existing implementations – see Sec. 1 in [@bauer2021ripser] for 
a review, and [@aggarwal2021dory; @vonbromssen2021computing] for recent entries 
not mentioned there.

Despite these successes, the computation of barcodes remains a challenge when
dealing with large datasets and/or with high-dimensional topological features. 
Indeed, the input to any barcode computation is a growing, one-parameter family 
of combinatorial objects, called a *filtration* or a *filtered complex*, and 
several filtrations of interest in applications quickly become very large as 
their defining parameter increases. This leads to a staggering number of 
elementary row or column operations required to distil the desired barcode via 
standard matrix reduction algorithms.

While PH computation for many types of simplicial filtrations constructed from 
point clouds, finite metric spaces, or graphs are limited by similar 
considerations, in this work we focus on *Vietoris–Rips* (VR) filtrations of 
finite metric spaces, as well as on *flag* filtrations of undirected graphs 
endowed with vertex and edge weights.

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
*Oineus* library [@nigmetov2020oineus]. Although optimizations such as clearing 
and implicit matrix reduction appear to have been implemented there, the code 
and performance are not optimized for Vietoris–Rips filtrations, and in 
particular no implementation of the emergent *or* apparent pair optimization is 
present there at this time. (At the time of writing, this library is in version 
1.0. We were not aware of its existence during the development of our code.)

*Ripser++* [@zhang2020gpuaccelerated] implements the idea of finding apparent 
pairs in parallel on a GPU to accelerate the computation of VR barcodes. 
Despite this, *Ripser++* is not fully parallel. For each dimension to process, 
it divides the computation into three sub-tasks: "*filtration construction and 
clearing*", "*finding apparent pairs*" and "*sub-matrix reduction*". The last 
of these steps is not parallel, and it is executed on the CPU. Hence, there 
is room for extending parallelism to the third sub-task above, which we try to 
harvest in this work by integrating the aforementioned ideas from 
[@morozov2020towards].

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
They compared [@glisse2022edge] their new version againt our. The achieved
outstading speed-ups, we hope at some point integrate their latest version.

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

*giotto-ph* is a library dedicated to the efficient computation of PH of VR 
filtrations (see \hyperref[sec:need]{``Statement of need"\ref*{sec:need}}). It 
inherits and extends ideas and code from many sources; \autoref{fig:lib} gives 
a visual representation of the most important ones among them. Our aim with 
*giotto-ph* is to provide an alternative to the excellent *Ripser.py* library, 
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

The main data structure of the algorithm described in [@morozov2020towards] 
is a lock-free hash table. A lock-free hash table is a concurrent hash map 
where concurrent operations do not make use of synchronization mechanisms 
such as mutexes. Instead, a lock-free hash table relies on atomic operations 
for manipulating its content; in particular, insertion is carried out by a 
mechanism called compare-and-swap (CAS). After benchmarking a few 
performance-oriented alternatives offering portability for most available 
compilers for Linux, Mac OS X, and Windows systems, we created a custom hash 
map adapted to the needs of the core matrix reduction algorithm, using the 
"Leapfrog" implementation in the open-source 
[*Junction*](https://github.com/preshing/junction) library.

As previously mentioned, we also adopted a different threading strategy: a 
*thread pool* (with optional CPU pinning option). A thread pool is a design 
pattern in which a "pool" of threads is created up front when the program 
starts, and the same threads are reused for different computations during 
the program's life span. This approach enables better amortization of the 
cost of the short-lived threads used in [@morozov2020lock], where one thread 
is created whenever needed and destroyed at the end of its computation task. 
\autoref{tbl:pool} compares the running time of a solution based on our thread
pool with the former approach. The run-time improvements are highly dataset 
dependent, but always measurable in the scenarios considered.

The final component in our C++ backend is a rewriting of the EC algorithm 
(see \hyperref[sec:related_work]{``Related work"\ref*{sec:related_work}}), 
implemented so far only in the *GUDHI* library [@gudhi:Collapse]. Our 
implementation focuses on performance and removes the dependencies on the 
*Boost* [@BoostLibrary] and *Eigen* [@eigenweb] libraries. It also supports 
weighted graphs with arbitrary (possibly non-positive) edge weights as well as 
arbitrary node weights. Improvements were achieved mainly by reworking data 
structures, making the implementation more cache-friendly, and directly 
iterating over data without any transformation, hence reducing the pressure on 
the memory sub-system.

\begin{table}
\centering
\caption{Running times, expressed in seconds, with and without the thread pool. 
$N$ denotes the number of threads used. All information regarding the datasets 
presented here is described in Section \ref{sec:experiments} and summarized in 
\autoref{tbl:datasets}.}
\label{tbl:pool}
\begin{tabular}{lrrrr}
    \hline
     & \multicolumn{2}{c}{\textbf{no thread pool}} & \multicolumn{2}{c}{\textbf{thread pool}} \\ \hline
    \textbf{dataset}    & $N = 8$ & $N = 48$ & $N = 8$ & $N = 48$ \\ \hline
    \texttt{sphere3}    & {0.4}   & {0.4}    & {0.4}   & 0.38     \\ \hline
    \texttt{dragon}     & {1.2}   & {1.2}    & {1.3}   & 1.3      \\ \hline
    \texttt{o3\_1024}   & {0.4}   & {0.18}   & {0.4}   & 0.17     \\ \hline
    \texttt{random16}   & {0.9}   & {0.4}    & {0.9}   & 0.24     \\ \hline
    \texttt{fractal}    & {0.9}   & {0.35}   & {0.9}   & 0.34     \\ \hline
    \texttt{o3\_4096}   & {6.9}   & {2.7}    & {6.9}   & 2.6      \\ \hline
    \texttt{torus4}     & {19}    & {14.7}   & {19.1}  & 14.3     \\
    \hline
\end{tabular}
\end{table}

## Python interface \label{sec:python}

Our Python interface is based on *Ripser.py* [@ctralie2018ripser]. While it 
lacks some of *Ripser.py*'s features, such as the support for "greedy 
permutations", it introduces the following improvements:

  - Support for Edge Collapser. EC is disabled by default, but users can easily 
    enable it by means of the `collapse_edges` optional argument.

  - Support for enclosing radius. The *(minimum) enclosing radius* of a 
    finite metric space is the radius of the smallest enclosing ball of that 
    space. Simplices with higher filtration values than the enclosing radius 
    can be safely omitted without changing the final barcode. When the 
    enclosing radius is considerably smaller than the maximum distance in the 
    data, doing so can lead to dramatic improvements in run-time and memory 
    usage [@henselmanghristl6, @henselmanpetrusek2020matroids]. Unless the user 
    specifies a threshold, *giotto-ph* makes use of the enclosing radius 
    optimization. An element of novelty in our interface is that, when both the 
    enclosing radius is computed and EC is enabled, the input distance 
    matrix/weighted graph is thresholded *before* being passed to the EC 
    backend. This can lead to substantial run-time improvements for the EC step.

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
resources, while ours (see \autoref{fig:scaling}) does not. The main culprit is
that, while in [@morozov2020lock] parallel resources are allocated only when 
needed in the computation, our thread pool (see 
\hyperref[sec:Cpp_backend]{``C++ backend"\ref*{sec:Cpp_backend}}) will allocate
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


