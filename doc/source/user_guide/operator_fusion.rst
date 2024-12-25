.. _operator_fusion:

===============
Operator Fusion
===============

Xorbits implements operator fusion optimization to reduce memory access overhead and improve computational efficiency. 
The fusion engine combines multiple nearby operators into a single fused one. Rather than implementing our own fusion 
engine from scratch, Xorbits leverages existing state-of-the-art fusion engines: NumExpr, JAX, or CuPy. Operator Fusion
is available automatically when one of the fusion packages is installed in your Python environment. Operator fusion is 
especially effective for ``xorbits.numpy``.

How It Works
-----------

The optimization process works as follows:

1. Identifies sequences of operations that can be fused together. 

Note that NumExpr, JAX, and CuPy are single-machine fusion engines, while Xorbits is a distributed toolkit. 
Xorbits will check which operations can be fused. For example, operations like single-axis reduction (``len(op.axis) == 1``
for ``xorbits.numpy.sum()`` or ``xorbits.numpy.max()``) can be fused, while other reduction operations are not.

2. Groups compatible operations into a single fused operation.

3. Executes the fused operation using the appropriate fusion engines (JAX, NumExpr, or CuPy).

This optimization reduces:

* Memory allocation/deallocation overhead
* Data movement between operations

The fusion can optimize chains of element-wise operations and simple reductions, 
where memory bandwidth is often the bottleneck rather than computational intensity.