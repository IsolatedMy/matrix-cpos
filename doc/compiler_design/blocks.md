# `block` Customization Point Design

<!-- vscode-markdown-toc -->
* [Motivation](#Motivation)
* [Construction](#Construction)
	* [BCSR Construction](#BCSRConstruction)
	* [BCSR View](#BCSRView)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='Motivation'></a>Motivation

Existing customization points include `row`, `column` and `diagonal`. Besides these, block view is known as a common view for parallelism. Therefore, `block` view is adopted as a new cp.

## <a name='Construction'></a>Construction
Here we implement BCSR format as a representative blocked sparse format and  each block is a `dense_matrix_view`. 

**BCSR**(Blocked Compressed Sparse Row) format is thought to be able to achieve resonable performance improvements compared to CSR with proper block size selected. Every block of BCSR is treated as a dense block, which may require padding with zeros. 

### <a name='BCSRConstruction'></a>BCSR Construction

It's recommended to use provided `mc::generate_bcsr` function to directly generate random benchmark by providing the following 5 arguments:
+ `m` and `n` describe the shape of matrix;
+ `block_height` and `block_width` describe the shape of block;
+ `nnz` specifies the number of non-zeros in sparse matrix;

```c++
auto [values, rowptr, colind, shape, a_nnz] =
    mc::generate_bcsr(m, n, block_height, block_width, nnz);
```

### <a name='BCSRView'></a>BCSR View

There are two ways to view generated BCSR matrix. First way is to use `bcsr_matrix_view`.

```c++
auto [values, rowptr, colind, shape, block_height, block_width, a_nnz] =
    mc::generate_bcsr(m, n, block_height, block_width, nnz);

mc::bcsr_matrix_view view(values, rowptr, colind,
                          shape, block_height, block_width, nnz);
```

The following 7 arguments are required to construct a BCSR view.
+ `values` is the array contains the entries of blocks from original matrix;
+ `rowptr` is the array contains the starting point of each row in block's view in `values` array;
+ `colind` is the array contains the column index of each block in block's view;
+ `shape` is the size of the original matrix;
+ `block_height` is the first dimension of block;
+ `block_width` is the second dimension of block;
+ `nnz` is the number of non-zero elements in original matrix.

In fact, `shape` and `nnz` can be deduced by other arguments. Currently, we choose to keep this redundancy.

For example, for matrix $A$ as follow:

$$
A = \left(
\begin{matrix}
0 & 2.42  & 0 & 0 & 0 & 0 \\
59.26 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 \\
85.34 & 91.42 & 82.82 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0\\
\end{matrix}
\right)
$$

Its `values`, `rowptr` and `colind` arrays are as follows:
```
values: [0, 2.42, 59.26, 0, 0, 0, 85.34, 91.42, 0, 0, 82.82, 0]
rowptr: [0, 1, 3, 3]
colind: [0, 0, 1]
```

Another way is to use `std::mdspan` to construct views for BCSR format. Compared with `bcsr_matrix_view` way, we only pass necessary argument here. Since conversion is user-invisible, it has more user-friendly interface `mc::blocks()`.

### Application: SpMV

<img src="fig/block-3.png" width=50%>

The following version is a static load balancing version of SpMV. Note the following points:
+ `Sch` represents the strategy for parallelizing;
+ The multiplication between block and vector $b$ need to compute the offset to add the result back to C. It's reasonable to compute offsets directly with `group_id` and `thread_id`, because we use static load balancing. In the following code, we assume `group_id` equals to the row base address of the block `x_base`, and `thread_id` equals to the column base address of the block `y_base`, as shown in the figure.
+ Currently, block and vector `b` and `c` are all stored in global memory.
+ `block.shape()` can be implemented as `block.width()` and `block.height()`.

```c++
template <block_iterable A, typename Sch>
void multiply(A&& a, B&& b, Sch&& S) {
 
  auto num_groups = 5;
  auto r = ndrange{num_groups, 512};
  auto a_blocks = a.blocks();

  auto balanced_blocks = static_load_balancer(a_blocks, num_groups, S);
 
  q.parallel_for(r, [=](auto id) {
    auto group_id = id.get_group().get_id();
    auto thread_id = id.get_local_id(0);
    
    if (group_id < balanced_blocks.size()) {
      auto blocks = balanced_blocks[group_id];
      if (thread_id < blocks.size()) {
        auto block = blocks[thread_id];
        
        // Treat block as dense_matrix_view
        for (std::size_t i = 0; i < block.shape()[0]; i ++) {
          for (std::size_t j = 0; j < block.shape()[1]; j ++) {
            auto x_addr = group_id + i;
            auto y_addr = thread_id + j;
            c[x_addr] += block[{i, j}] * b[y_addr];
          }
        }
      }
    }
  }).wait();
}
```

The following is a dynamic load balancing version of SpMV. Compared with static load balancing version, the following points need to be noted:
+ Since multiple threads can request block from `balanced_blocks`, I think it's necessary to set the read of `balanced_blocks` as atomic operation.

```c++
template <block_iterable A, typename Sch>
void multiply(A&& a, B&& b, Sch&& S) {
 
  auto num_groups = 5;
  auto r = ndrange{num_groups, 512};
  auto a_blocks = a.blocks();

  auto balanced_blocks = dynamic_load_balancer(a_blocks, num_groups, S);
 
  q.parallel_for(r, [=](auto id) {
    auto group_id = id.get_group().get_id();
    auto thread_id = id.get_local_id(0);
    
    atomic_ref<...> atomic_data(balanced_blocks);
    auto&& blockZip = balanced_blocks.pop();
    if (blockZip) {
      auto x_base = blockZip.base()[0];
      auto y_base = blockZip.base()[1];
      auto block = blockZip.value();
      for (std::size_t i = 0; i < block.shape()[0]; i ++) {
        for (std::size_t j = 0; j < block.shape()[1]; j ++) {
          auto x_addr = x_base + i;
          auto y_addr = y_base + j;
          c[x_addr] += block[{i, j}] * b[y_addr];
        }
      }
    }
  }).wait();
}
```