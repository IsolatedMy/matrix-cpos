# `block` Customization Point Design

<!-- vscode-markdown-toc -->
* [Motivation](#Motivation)
* [Construction](#Construction)
	* [BCSR Construction](#BCSRConstruction)
	* [BCSR View](#BCSRView)
* [Design](#Design)
	* [SpMV](#SpMV)
		* [Distribution techniques](#Distributiontechniques)
		* [Static Load Balancer](#StaticLoadBalancer)

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

## <a name='Design'></a>Design
In this section, we introduce the design idea for SpMV and SpMM through the interfaces we designed. After that, we provide details of our interfaces.

### <a name='SpMV'></a>SpMV

**Sparse matrix-vector multiplication (SpMV)** of the form $y = Ax$ is a widely used computational kernel existing in many scientific applications. The input matrix $A$ is sparse. The input vector $x$ and the output vector $y$ are dense.

<img src="fig/block-3.png" width=50%>

#### <a name='Distributiontechniques'></a>Distribution techniques

In this part, we will provide three distribution technique versions of SpMV based on [this article](https://www.ibm.com/docs/en/pessl/5.3.0?topic=distributions-distribution-techniques) from IBM:

The following code is the version of SpMV using **block distribution**. 
+ The details of `static_load_balancer` is put aside in this part;
+ Memory transfer is not considered here. The data are all stored in global memory.

```c++
template <block_iterable A, random_access_range B, 
          random_access_range C>
void multiply(A&& a, B&& b, C&& c) {
  auto M = 32;
  auto N = 32;
  auto m = 2;
  auto n = 2;
  auto r = ndrange{{M, N}, {m, n}};

  auto balanced_blocks = static_load_balancer(a.row_blocks(), {M, N});
 
  q.parallel_for(r, [=](auto id) {
    auto group_id_x = id.get_global_id()[0];
    auto group_id_y = id.get_global_id()[1];

    auto local_id_x = id.get_local_id()[0];
    auto local_id_y = id.get_local_id()[1];
    
    if (balanced_blocks[{group_id_x, group_id_y}]) {
      auto blocks = balanced_blocks[{group_id_x, group_id_y}];

      // Block distribution
      auto group_size = m * n;
      auto n_tasks = ceil(blocks.size() / group_size);
      auto id = local_id_x * n + local_id_y;
      for (auto ib = id * n_tasks; ib < min((id+1)*n_tasks, blocks.size()); ib ++) {

        auto blockZip = blocks[ib];
        auto&& [x_base, y_base] = std::get<0>(blockZip);
        auto block = std::get<1>(blockZip);

        for (std::size_t i = 0; i < block.shape()[0]; i ++) {
          for (std::size_t j = 0; j < block.shape()[1]; j ++) {
            auto x_addr = x_base + i;
            auto y_addr = y_base + j;
            c[x_addr] += block[{i, j}] * b[y_addr];
          }
        }
      }
    }
  }).wait();
}
```

The **cyclic distribution** code is as follows
```c++
if (balanced_blocks[{group_id_x, group_id_y}]) {
  auto blocks = balanced_blocks[{group_id_x, group_id_y}];

  // Cyclic distribution
  auto group_size = m * n;
  auto id = local_id_x * m + local_id_y;
  for (auto ib = id; ib < blocks.size(); ib += group_size) 
  
  /// ...
}
```

The **block-cyclic distribution** code is as follows
```c++
if (balanced_blocks[{group_id_x, group_id_y}]) {
  auto blocks = balanced_blocks[{group_id_x, group_id_y}];

  // Block-cyclic distribution
  auto group_size = m * n;
  auto r = 2;   // Block size for each work-item
  auto id = local_id_x * m + local_id_y;
  for (auto ib_base = id; ib_base < blocks.size(); ib_base += group_size * r) {
    for (auto ib = ib_base; ib < ib_base + r; ib_base++ ) {
        /// ...
    }
  }
}
```

#### Load balancer

Static block-distribution balancer.

```c++
// Number 1: static block distribution balancer
template <std::ranges::random_access_range R>
auto static_load_balancer(R&& range, size_t num_chunks) {
  auto chunk_size = ceil((float)range.size() / num_chunks)
  auto balanced_chunks = range | std::views::chunk(chunk_size);
  return balanced_chunks;
}

/*
 * Example input/output
 * range: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 * num_chunks: 2
 * output: [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
 * 
 * num_chunks: 5
 * output: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
*/
```

Static block-cyclic-distribution balancer.

```c++
// Number 2: static blcok cyclic distribution balancer
template <std::range::random_access_range R>
auto static_load_balancer(R&& range, size_t num_ranges, size_t chunk_size) {
  auto chunks = range | std::views::chunk(chunk_size);
  auto range_size = ceil((float)chunks.size() / num_ranges);
  auto balanced_groups = chunks | std::views::chunk(range_size);
  return balanced_groups;
}

/*
 * Example input/output
 * range: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 * num_range: 2
 * chunk_size: 3
 * output: [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9]]]
 * 
 * Worker 1 sees: [[0, 1, 2], [3, 4, 5]]
 * Worker 2 sees: [[6, 7, 8], [9]]
 * 
 * P.S. Actually, in block-cyclic distribution
 * Worker 1 sees: [[0, 1, 2], [6, 7, 8]]
 * Worker 2 sees: [[3, 4, 5], [9]]
*/

template <std::range::random_access_range R>
auto static_load_balancer(R&& range, size_t num_ranges, size_t chunk_size) {
  auto chunks = range | std::views::chunk(chunk_size);
  auto group_ids = __ranges::views::iota(I(0), I(num_ranges));
  auto balanced_groups = group_ids | __ranges::views::transform(
    [=](auto group_id) {
      return __ranges::subrange(chunks + group_id, ranges::end(chunks)) | stride(num_ranges);
    }
  );
  return balanced_groups;
}
```

Dynamic balancer.

```c++
template <std::ranges::random_access_range R>
auto dynamic_load_balancer(R&& range, size_t num_ranges, size_t chunk_size) {
}

/*
 * Example input:
 * range: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 * num_ranges: 2
 * job_size: 2
 * output [(*load_balance_queue object*), (load_balance_queue object)]
 * 
 * Worker 1 sees: [[0, 1], [6, 7], ...]
 * Worker 2 sees: [[2, 3], [4, 5], [8, 9], ...]
 * 
*/
```