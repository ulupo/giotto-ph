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
    orcid: XXX
    equal-contrib: true
    affiliation: 1
  - name: Sydney Hauke
    orcid: XXX
    equal-contrib: true
    affiliation: 1
  - name: Umberto Lupo
    orcid: 0000-0001-6767-493X
    equal-contrib: true
    corresponding: true
    affiliation: "2, 3"
  - name: Matteo Caorsi
    orcid: XXX
    affiliation: 4
  - name: Alberto Dassatti
    orcid: XXX
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

# Statement of need {#sec:need}

In recent years, *persistent homology* (PH) (see e.g. [@ghrist2007barcodes; 
@edelsbrunner2008persistent; @edelsbrunner2014persistent; 
@oudot2015persistence; @chazal2016structure; @perea2018brief; 
@carlsson2019persistent; @nanda2021computational] for surveys) has been a key 
driving force behind the ever-increasing adoption of topological approaches in 
a wide variety of computational contexts, such as geometric inference 
[@edelsbrunner2014short; @boissonnat2018geometric], signal processing 
[@robinson2014topological; @perea2015sliding], data visualization 
[@tierny2018topological], and, more generally, data analysis 
[@carlsson2009topology; chazal2021introduction] and machine learning 
[@hensel2021survey]. Among the main invariants described by this theory, the 
(*persistence*) *barcode* [@frosini1990shapes; @frosini1992measuring; 
@barannikov1994morse; @robins1999approximations; 
@edelsbrunner2000simplification; @zomorodian2005computing] has attracted the 
most attention due to (a) its ability to track the appearance and 
disappearance of topological features in data throughout entire ranges of 
parameters, (b) its succinct nature and ease of representation, as it simply 
consists of a (typically small) collection of intervals of the real line, (c) 
its provable robustness under perturbations of the input data 
[@damico2003optimal; cohen-steiner2007stability], and (d) its amenability to 
computation and algorithmic optimization, as demonstrated by the large 
number of existing implementations – see Sec. 1 in [@bauer2021ripser] for a 
review, and [@aggarwal2021dory; vonbromssen2021computing] for recent entries 
not mentioned there.

Despite these successes, the computation of barcodes remains a challenge 
when dealing with large datasets and/or with high-dimensional topological 
features. We now explain why this is the case. The input to any barcode 
computation is a growing, one-parameter family of combinatorial objects, 
called a *filtration* or a *filtered complex*. Filtrations consist of 
*cells* with assigned integer *dimensions* and values of the 
filtration-defining parameter, as well as *boundary* (resp. *co-boundary*) 
relations (mappings) between $k$-dimensional cells and ($k-1$)-dimensional 
[resp. ($k+1$)-dimensional] ones (we refer the reader to any of the 
aforementioned surveys of PH for rigorous definitions). Arguably, the most
common examples of filtrations in applications concern *simplicial* 
complexes, in which case the cells are referred to as *simplices*, and 
$k$-dimensional simplices consist of sets of $k + 1$ points from a common 
vertex set $V$ – for instance, a $0$-simplex $\{v\}$ is one of the vertices 
$v \in V$, while a $1$-simplex $\{v, w\} \subseteq V$ can be thought of as 
an edge connecting vertices $v$ and $w$. These are the filtrations of 
interest in the present paper. In particular, we focus on the 
*Vietoris–Rips* (VR) (resp. *flag*) filtration of a finite metric space 
(resp. undirected graph with vertex and edge weights), in which simplices 
are arbitrary subsets of the available points (resp. vertices) and their 
filtration values are set to be their diameters (resp. the maximum weights 
of all vertices and edges they contain, with absent edges being given 
infinite filtration value).

Several simplicial filtrations of interest in applications, and the VR
filtration chiefly among these, quickly become very large as their defining 
parameter increases (and hence more and more simplices are included in the 
growing complex). At the heart of all algorithms for computing PH barcodes 
lies the reduction of *boundary* or *co-boundary matrices* indexed by the 
full set of simplices in the filtration; the available reduction algorithms 
have asymptotic space and time complexities which are polynomial in the total 
number $N$ of simplices. In the case of the VR filtration of an input metric 
space $\mathcal{M}$, if one is interested in computing the barcode up to and 
including homology dimension $D$, then $N = \sum_{k=0}^{D + 1} \binom{|M|}{k}$. 
For sizeable datasets, this combinatorial explosion leads to a staggering number 
of elementary row or column operations (as well as memory) required to distil 
the desired barcode. PH computation for many other simplicial filtrations 
constructed from point clouds, finite metric spaces, or graphs are also 
ultimately limited by similar considerations.

# Related work {#sec:related_work}

To the best of our knowledge, at the time of writing 
[*Ripser*](https://github.com/Ripser/ripser) [@bauer2021ripser] is the *de 
facto* state of the art and reference for computing VR persistence barcodes 
on CPUs. *Ripser* uses multiple known optimizations like *clearing* 
[@chen2011persistent] and *cohomology* [@desilva2011dualities]. Furthermore, 
it makes use of other performance-oriented ideas, such as the implicit 
representation of the (co)boundary and reduced (co)boundary matrices, and 
the *emergent/apparent pairs* optimizations (we refer to [@bauer2021ripser] 
for definitions and details). At the time of writing, the latest version of 
*Ripser* is *v1.2* (release date: 25 February 2021).  In that version, to 
the emergent pairs optimization in use until that point was added an 
optimization based on apparent pairs.

Although *Ripser v1.2* is arguably the fastest existing code for computing VR
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

Zhang *et al.*'s *Ripser++* [@zhang2020gpuaccelerated] implements the idea of 
finding apparent pairs in parallel on a GPU to accelerate the computation of VR 
barcodes. Despite this, *Ripser++* is not fully parallel. For each dimension to 
process, it divides the computation into three sub-tasks: 
"*filtration construction and clearing*", "*finding apparent pairs*" and 
"*sub-matrix reduction*". The last of these steps is not parallel, and it is 
executed on the CPU. According to 
[Amdahl’s law](https://en.wikipedia.org/wiki/Amdahl's\_law) this processing 
sequence is expected to yield only diminishing returns when augmenting the 
number of parallel resources. Although performance gains have been demonstrated 
in [@zhang2020gpuaccelerated] – particularly when using high-end GPUs – there 
is room for extending parallelism to the third sub-task above, which we try to 
harvest in this work by integrating the aforementioned ideas from 
[@morozov2020towards].

All implementations presented in this subsection (barring [@nigmetov2020oineus], 
which also provides some Python bindings) are written in low-level languages 
(C++, CUDA). However, researchers today make wide use of higher-level languages 
– Python, for instance, is the dominant one in several fields. It is therefore 
natural that libraries have been developed to couple Python's ease of use with 
the high performance provided by these pieces of code. *Ripser.py* 
[@ctralie2018ripser] is probably the most notable of these implementations, 
providing an intuitive interface for VR filtrations wrapping *Ripser* at its 
core. The authors of the library forked the original *Ripser* implementation 
and added support for non-zero birth times, as well as the possibility to 
compute and retrieve cocycles.

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

# Our contribution

In this context, we present 
[*giotto-ph*](https://github.com/giotto-ai/giotto-ph), a Python package built 
on top of a C++ backend that computes PH barcodes for VR filtrations on the CPU. 
To the best of our knowledge, this is the first package fully integrating the 
three ideas described in [@sec:related_work] (lock-free reduction, 
parallelized search for apparent pairs, edge collapses) in a single portable, 
easy-to-use library. We remark that, after the release of our code and of the 
first version of this paper, we learned about a very recent thesis 
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
[@sec:python] for details). The figure also shows the 
inheritance of *giotto-ph*'s C++ backend from pre-dating implementations.
\label{fig:lib}](architecture_bpj.svg){width=100%}

*giotto-ph* is a library dedicated to the efficient computation of PH of VR 
filtrations (see [@sec:need]). It inherits and extends ideas and 
code from many sources; \autoref{fig:lib} gives a visual representation of the 
most important ones among them. Our aim with *giotto-ph* is to provide an 
alternative to the excellent *Ripser.py* library, retaining several of the 
latter's advantages, namely portability and ease of use, while replacing the 
C++ backend with a new parallel and higher-performance version. 

## C++ backend {#sec:Cpp_backend}

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
\autoref{tbl:pool} compares the running time of a solution based on our 
thread pool with the former approach. The run-time improvements are highly 
dataset dependent, but always measurable in the considered scenarios.

: Comparison of programming languages used in the publishing tool.
  []{label="proglangs"}

\autoref{proglangs}

| Language | Typing          | Garbage Collected | Evaluation | Created |
|----------|:---------------:|:-----------------:|------------|---------|
| Haskell  | static, strong  | yes               | non-strict | 1990    |
| Lua      | dynamic, strong | yes               | strict     | 1993    |
| C        | static, weak    | no                | strict     | 1972    |



The final component in our C++ backend is a rewriting of the EC algorithm 
(see [@sec:related_work]), implemented so far only in the *GUDHI*
library [@gudhi:Collapse]. Our implementation focuses on performance and removes 
the dependencies on the *Boost* [@BoostLibrary] and *Eigen* [@eigenweb] 
libraries. *giotto-ph*'s EC is more than 1.5 times faster than the original 
version as reported in [@tbl:collapser]. It also supports weighted 
graphs with arbitrary (possibly non-positive) edge weights as well as 
arbitrary node weights. Improvements were achieved mainly by reworking data 
structures, making the implementation more cache-friendly, and directly 
iterating over data without any transformation, hence reducing the pressure 
on the memory sub-system.

::: []
                  **giotto-ph backend**                                   
  ------------- ----------------------- --------------- ----------------- ---------------
                     **no thread pool**                   **thread pool** 
  **dataset**                   $N = 8$       $N = 48$            $N = 8$        $N = 48$
  `sphere3`                         0.4             0.4               0.4            0.38
  `dragon`                          1.2             1.2               1.3             1.3
  `o3_1024`                         0.4            0.18               0.4            0.17
  `random16`                        0.9             0.4               0.9            0.24
  `fractal`                         0.9            0.35               0.9            0.34
  `o3_4096`                         6.9             2.7               6.9             2.6
  `torus4`                           19            14.7              19.1            14.3

  : Running times, expressed in seconds, with and without the thread pool. $N$
  denotes the number of threads used. All information regarding the datasets 
  presented here are described in [@sec:experiments] and summarized in 
  [@tbl:datasets].[]{label=tbl:pool}
:::

## Python Interface {#sec:python}

Our Python interface is based on *Ripser.py* [@ctralie2018ripser]. While it 
lacks some of *Ripser.py*'s features, such as the support for "greedy 
permutations" and for retrieving cocycles, it introduces the following 
notable improvements:

  - Support for Edge Collapser. EC is disabled by default because it is 
    expected and empirically confirmed that, unless the data is large and/or 
    the maximum homology dimension to compute is high, the initial run-time 
    overhead due to EC is often not compensated for by the resulting 
    speed-up in the downstream reduction steps (see the end of 
    [@sec:related_work]). However, users can easily enable it by means of the 
    `collapse_edges` optional argument. In Table ??? we will 
    show the difference in run-times when this option is active. See also 
    "Support for enclosing radius", below.

  - Support for enclosing radius. The *(minimum) enclosing radius* of a 
    finite metric space is the radius of the smallest enclosing ball of that 
    space. Its computation, starting from a distance matrix, is trivial to 
    implement and takes negligible run-time on modern CPUs. Above this 
    filtration value, the Vietoris–Rips complex becomes a cone, and hence 
    all homology groups are trivial. Hence, simplices with higher filtration 
    values than the enclosing radius can be safely omitted from the 
    enumeration and matrix reduction steps, without changing the final 
    barcode. When the enclosing radius is considerably smaller than the 
    maximum distance in the data, this can lead to dramatic improvements in 
    run-time and memory usage, as observed in 
    [@henselmanpetrusek2020matroids]. (For instance, the barcode 
    computation for the `random16` dataset (see [@tbl:datasets]) up 
    to dimension $7$ would not be completed after two hours without the 
    enclosing radius optimization; with it, the run-time drops to seconds. 
    Not all datasets can be expected to witness equally impressive 
    improvements, but the cost of computing the enclosing radius is trivial 
    compared to the computation of PH.)  Unless the user specifies a 
    threshold, both *Eirene* [@henselmanghristl6] and *Ripser* make use of 
    the enclosing radius optimization, and the same is true in *giotto-ph*, 
    where the enclosing radius computation is implemented in Python using 
    highly optimized *NumPy* functions. An element of novelty in our 
    interface is that, when both the enclosing radius is computed and EC is 
    enabled, the input distance matrix/weighted graph is thresholded 
    *before* being passed to the EC backend. As we experimentally find and 
    report in [@sec:collapser], on several datasets this can 
    lead to substantial run-time improvements for the EC step.

  - Weighted VR filtrations. While standard stability results for VR barcodes 
    [@cohen-steiner2007stability; @chazal2009proximity] guarantee robustness 
    to small perturbations in the data, VR barcodes are generally *unstable* 
    with respect to the insertion or deletion of even a single data point. 
    Thus, even relatively small changes in the local density can greatly 
    affect the resulting barcodes, rendering the vanilla VR persistence 
    pipeline very vulnerable to statistical outliers. Distance-to-measure 
    (DTM) based filtrations [@anai2020dtmbased] address this issue by 
    re-weighting vertices and distances according to the local neighbourhood 
    structure. The user can toggle DTM-based reweighting (or more general 
    reweightings) by appropriately setting the optional parameters `weights` 
    and `weight_params`.

  - [*pybind11*](https://github.com/pybind/pybind11) bindings. We added 
    support for and used *pybind11* instead of [*Cython*](https://cython.org/) 
    for creating Python bindings. In our experience, it is easier to use without 
    compromising performance. Furthermore, it is already used for the bindings 
    in the *giotto-tda* library [@tauzin2021giottotda], our sibling project. 
    The presence of Python bindings as well as the portability on different 
    operating systems, namely Linux, Mac OS X, and Windows, have been two of 
    our core objectives to facilitate the adoption of our library.

# Experimental results {#sec:experiments}

All experiments presented in this paper were performed on a machine running 
Linux CentOS 7.9.2009 with kernel 5.4.92, equipped with two Intel® XEON® 
Gold 6248R (24 physical cores each) and a total of 128 GB of RAM. 

We present measures on the datasets of [@tbl:datasets] because they 
are publicly available, and they are used in publications [@Otter_2017; 
@bauer2021ripser] describing established algorithms, making them a 
representative benchmark set and facilitating comparisons among competing 
solutions. All datasets are stored as point clouds. When the `threshold` 
parameter is empty, the tests report run-times with the enclosing radius 
option active. The `dim` parameter corresponds to the maximum dimension for 
which we compute PH, and the `coeff` parameter corresponds to the prime 
field of coefficients (in our tests, this is always $\mathbb{F}_2$).

::: {#tbl:datasets}
  **dataset**     **size** **threshold**     **dim**   **coeff**
  ------------- ---------- --------------- --------- -----------
  `sphere3`            192                         2           2
  `dragon`            2000                         1           2
  `o3_1024`           1024 1.8                     3           2
  `random16`            50                         7           2
  `fractal`            512                         2           2
  `o3_4096`           4096 1.4                     3           2
  `torus4`           50000 0.15                    2           2

  : Datasets used for benchmarking. "Size" means the number of points in
  the dataset.
:::

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

\autoref{fig:scaling} shows the scaling of *giotto-ph* when increasing 
the number of worker threads. Scaling is different for each dataset due to 
the variable number of apparent and emergent pairs as well as the dimension 
parameter used. Observe that the larger the number of points in the dataset, 
the better the scaling. Similar effects are visible for datasets in which 
when the maximal homology dimension to compute is higher.

As seen above for run-times versus *Ripser v1.2*, for scaling, too, the least 
favourable results are obtained on datasets such as `sphere3` and `sphere3`, 
in which both the size and maximum homology dimension to compute are small.

![Scaling of *giotto-ph* when increasing the number of threads. This figure 
is similar to \autoref{fig:gph_vs_ripser_1.2} because in a single thread 
configuration *giotto-ph* performs very similarly to *Ripser v1.2*.
\label{fig:scaling}](giotto_speedup.svg){width=80%}

According to our measurements reported in Figure \autoref{fig:moro}, our 
implementation outperforms Morozov and Nigmetov's proof-of-concept 
implementation [@morozov2020lock] in most cases, and most noticeably when 
the number of parallel resources increases. The only exception when using 
multiple threads is `sphere3`. The version in [@morozov2020lock] performs 
better and better on `sphere3` when increasing the number of parallel 
resources, while ours (see \autoref{fig:scaling}) does not. The main culprit is
that, while in [@morozov2020lock] parallel resources are allocated only when 
needed in the computation, our thread pool (see [@sec:Cpp_backend]) will 
allocate all the parallel resources indicated by the user ahead of time. Our 
approach is most beneficial when the allocated resources can be reused during 
the computation, and this is true e.g. when computing homology dimensions in 
degree $2$ and above. However, when computing only up to dimension $1$, it is 
only necessary to allocate the parallel resources once, and an on-the-fly 
approach such as the one in [@morozov2020lock] can be faster. Another logically 
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

## Higher homology dimensions

Table ??? compares *Ripser v1.2* and *giotto-ph* when
increasing the homology dimension parameter. We included the measurements
using EC to show the potential benefits. It is important to note that
timings reported using EC do not include EC processing time; the interested
reader can find them in [@tbl:collapser]. The first dimension reported in 
Table ??? is the one in the setup of [@tbl:datasets].

`sphere3` and `random16` are the only datasets where the Maximal Index (**MI**)
(i.e. the maximum number of retrievable entries) is not attained. `sphere3`
is a highly regular dataset and computing higher homology dimensions will
not yield interesting results. `random16` produces no barcodes at dimension
$20$. We arbitrarily decided to stop at dimension $10$ and report the data.

[@tbl:datasets] shows that, in general, pre-processing with EC
leads to a reduction in later run-times. The only exception is the `sphere3`
dataset, where EC is slightly detrimental. The reason for this is
implementational in nature as we now explain. The EC step takes as input the
dataset's distance matrix in dense format, and outputs a sparse matrix. In
the case of `sphere3`, EC removes very few edges, producing a highly filled
sparse matrix. Dense representations have better cache behaviour than sparse
ones, and thus can lead to faster computations than highly filled sparse
ones. We are working on a heuristic to automatically select the best data
format.

## Edge Collapser {#sec:collapser}

We now report experimental findings concerning our EC implementation. These 
are summarized in [@tbl:collapser], where the third column 
demonstrates that our solution is always faster than *GUDHI*'s original one 
on the datasets considered.

We remind the reader that, as explained in [@sec:python], a 
novelty of our implementation is the use of the enclosing radius computation to
shorten the run-time of the EC step even beyond what is already made possible 
by our use of faster routines and data structures. The experimental impact of 
this enhancement is shown in the last column of [@tbl:collapser]. One would 
expect that the more "random" datasets, where "central points" are likely to be 
present, will benefit the most from thresholding by the enclosing radius. Among 
our standard datasets from [@tbl:datasets], `random16`, `o3_1024` and `o3_4096` 
are random datasets, but we do not witness such an impact. While, in the case 
of `random16`, the reason is likely that the dataset it too small ($50$ 
points), in the case of the `o3` datasets the reason is that a threshold lower 
than the enclosing radius is provided, meaning that the enclosing radius 
optimization is not used at all there. To demonstrate that our expectation is 
valid despite the limitations caused by our choice of datasets and 
configurations, we have added an entry to [@tbl:datasets], representing a 
dataset of $3000$ points sampled from the uniform distribution on the unit cube 
in $\mathbb{R}^3$. Together with `sphere3`, this example shows that large gains 
can be made by using the enclosing radius on certain datasets.

::: {#tbl:collapser}
  **dataset**     ***GUDHI* EC**   ***giotto-ph* EC** (speedup)   ***giotto-ph* EC with encl. rad.** (speedup) 
  ------------- ---------------- ------------------------------ ---------------------------------------------- --
  `sphere3`                  1.6                     0.9 (1.78)                                     0.9 (1.78) 
  `dragon`                    63                      36 (1.75)                                      28 (2.25) 
  `o3_1024`                  0.2                    0.13 (1.53)                                  0.13\* (1.53) 
  `random16`               0.004                   0.001 (4.00)                                   0.001 (4.00) 
  `fractal`                 1.32                     0.8 (1.65)                                     0.8 (1.65) 
  `o3_4096`                  2.1                     1.2 (1.75)                                   1.2\* (1.75) 
  `torus4`                    10                     6.7 (1.49)                                   6.7\* (1.49) 
                             180                     125 (1.44)                                      78 (2.31) 

  : Run-time comparison between *GUDHI*'s implementation
  [@gudhi:Collapse] of the EC algorithm of Boissonnat and Pritam
  [@boissonnat2020edge] and *giotto-ph*'s implementation. The last
  column reports run-times when sparsifying according to the enclosing
  radius before calling *giotto-ph*'s EC, which is the default behaviour
  when no threshold is provided by the user. All execution times are in
  seconds, while speedups are ratios. Cells marked with an asterisk mean
  that a threshold is provided and therefore the enclosing radius is not
  computed by default. The last entry is unique to this table and better
  demonstrates the impact of the enclosing radius optimization on
  favourable datasets and configurations.
:::

# Conclusion and future work

We integrated multiple, existing and novel, algorithmic ideas to obtain a 
state-of-the-art implementation of the computation of persistent homology for 
Vietoris–Rips filtrations. This implementation enables the use of parallel CPU 
resources to speed up the computation and outperforms even state-of-the-art GPU 
implementations.

We plan to extend *giotto-ph* by supporting a wider range of filtrations in a 
modular way. We also plan to add features (e.g., simplex pairs and essential 
simplices) needed for back-propagation in a deep learning context, and seamless 
integration with frameworks such as [*PyTorch*](https://pytorch.org/).

# Acknowledgements

We thank Anibal Medina-Mardones and Kathryn Hess Bellwald for numerous fruitful 
discussions, as well as Ulrich Bauer for very helpful conversations about 
\textit{Ripser}. This work was supported by the Swiss Innovation Agency 
(Innosuisse project 41665.1 IP-ICT).

# References


