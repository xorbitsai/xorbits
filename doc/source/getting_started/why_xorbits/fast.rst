.. _why_xorbits_fast:

Why is Xorbits so fast?
===============================

Xorbits is the fastest compared to other popular frameworks according to our benchmark tests. See our
performance comparison with other frameworks :ref:`here <xorbits_performance_comparison_index>`.

Architectural Decisions
-----------------------

Xorbits was initially designed with the purpose of boosting the performance
of pandas on either a single machine or a cluster. Our primary goal is to
enhance scalability and performance. The Xorbits team has made a series 
of architectural decisions that collectively enabled us to accomplish this goal:

**No GIL any more!** To break through the limitations of Python GIL, a single machine 
accelerates computing through multiple processes, and multiple processes communicate 
through shared memory, reading data from other processes is almost zero-copy. 

**Dynamic partitioning:** Xorbits can transparently reshape the partitioning as required for the 
corresponding operation, depending on whether the operation is row-parallel or column-parallel. 
This allows Xorbits to support more of the pandas API (e.g., :code:`transpose`, :code:`median`,
:code:`quantile`) and leverage more parallelism. 

**Streaming datasets:** Xorbits streams in the data into memory only the data required for
computations, while the remainder can be stored on the disk/cloud storage. This method is
efficient and saves memory resources.

**Scalability:** Xorbits can leverage all available CPU cores and disks to execute even a 
single query. Not only on a single server but all CPU cores and disks of a cluster as well.

**Indexes:** Memory resident Xorbits data structures allow the reading of only the necessary 
columns required for computations, and only the necessary row ranges of those columns.

**Hardware-accelerated:** In situations where multiple workers are communicating, 
Xorbits takes advantage of the best available hardware to speed up data transmission, such as 
InfiniBand, NVLink, and so on.

Optimization Techniques
-----------------------

What sets Xorbits apart is our dedication to meticulous attention to detail and the application 
of numerous optimization techniques.

Graph-based Plan Optimizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Xorbits incorporates the concept of a computational graph as an abstraction. Leveraging the computational
graph, Xorbits can perform many optimizations on the execution plan before actual computation. A series
of optimization rules such as **Column Pruning**, **Predicate Pushdown** and **Operator Fusion** can make
the computation faster and use less memory. 

Graph Fusion Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

Xorbits proposes a graph fusion algorithm based on graph coloring. The coloring algorithm will determine
the initial number of colors according to the computational resources. According to the topological
structure of the graph, it finds computational nodes that can be merged into a subgraph as the smallest
unit of execution. It can simplify the computational graph and optimize execution at runtime stage using
libraries, e.g. `JAX <https://github.com/google/jax>`__ and `NumExpr <https://github.com/pydata/numexpr>`__,
or JIT optimization techniques. 

.. image:: /_static/graph_fusion.png
   :alt: Graph fusion optimization
   :align: center
   :scale: 35%

Adaptive Execution Planning
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Xorbits features a planner capable of adaptively generating execution plans. For example, in a
:code:`groupby.agg` calculation, the planner will check the compression ratio of the result data after
processing some data. If a high degree of duplicate values results in high data compression, a tree-structured
computational graph is employed. However, if the compression ratio is low, potentially leading to Out-of-Memory
issues with the tree-structured method, the planner automatically opts for a shuffle approach to perform the
calculation. This adaptive strategy is applied to numerous operators, enhances the efficiency of our computations. 

The following figure shows how we parallelize :code:`groupby.agg()`:

.. image:: /_static/adaptive_plan.png
   :alt: Adaptive execution planning
   :align: center
   :scale: 45%