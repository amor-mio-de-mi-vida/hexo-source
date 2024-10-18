---
title: Tensor Program Abstraction
tags:
  - course
categories:
  - MLCompiler
date: 2024-10-14 21:32:28
date modified: 2024-10-18 15:44:57
---
## Primitive Tensor Function

Primitive Tensor Function a tensor function that corresponds to a single "unit" of computational operation.


## Tensor Program Abstraction


- **Multi-dimensional buffers**: that holds the input, output, and intermediate results.

- **Loop nests**: that drive compute iterations.

- **Computations**: statement.


Primitive tensor function refers to the single unit of computation in model execution. 

- One important MLC process is to transform implementation of primitive tensor functions

Tensor program is an effective abstraction to represent primitive tensor functions.

- Key elements include: multi-dimensional buffer, loop nests, computation statement.

- Program-based transformations can be used to optimize tensor programs.

- Extra structure can help to provide more information to the transformations.


implementation of `mm-rel`u in low level
```python
def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
	Y = np.empty((128, 128), dtype="float32")
	for i in range(128):
		for j in range(128):
			for k in range(128):
			 if k == 0:
				 Y[i, j] = 0
			Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
	for i in range(128):
		for j in range(128):
			C[i, j] = max(Y[i, j], 0)
```

Key features: 

- Multi-dimensional buffer (arrays).

- Loops over array dimensions.

- Computations statements are executed under the loops

The code below shows a TensorIR implementation of `mm_relu`. The particular code is implemented in a language called TVMScript, which is a domain-specific dialect embedded in python AST.

```python
@tvm.script.ir_module
class MyModule:
	@T.prim_func
	def mm_relu(A: T.Buffer[(128, 128), "float32"],
				B: T.Buffer[(128, 128), "float32"],
				C: T.Buffer[(128, 128), "float32"]):
		T.func_attr({"flobal_symbol": "mm_relu", "tir.noalias": True})
		Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
```

### Transformation

```python
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j0 in range(32):
            for k in range(128):
                for j1 in range(4):
                    j = j0 * 4 + j1
                    if k == 0:
                        Y[i, j] = 0
                    Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)

c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu_v2(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)
```

