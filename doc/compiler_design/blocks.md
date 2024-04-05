# `block` Customization Point Design

<!-- vscode-markdown-toc -->
* [Motivation](#Motivation)
* [Construction](#Construction)
	* [BCSR Construction](#BCSRConstruction)
	* [Using MdSpan](#UsingMdSpan)
* [Application](#Application)
	* [SpMV](#SpMV)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='Motivation'></a>Motivation

Existing customization points include `row`, `column` and `diagonal`. Besides these, block view is known as a common view for parallelism. Therefore, `block` view is adopted as a new cp.

## <a name='Construction'></a>Construction
Here we implement BCSR format as a representative blocked sparse format and  each block is a `dense_matrix_view`. 

### <a name='BCSRConstruction'></a>BCSR Construction

It's recommended to use provided `mc::generate_bcsr` function to directly generate random benchmark by providing the following 5 arguments:
+ `m` and `n` describe the shape of matrix;
+ `block_height` and `block_width` describe the shape of block;
+ `nnz` specifies the number of non-zeros in sparse matrix;

```c++
auto [values, rowptr, colind, shape, a_nnz] =
    mc::generate_bcsr(m, n, block_height, block_width, nnz);
```

### BCSR View

There are two ways to view generated BCSR matrix. First way is to use `bcsr_matrix_view`.

```c++
auto [values, rowptr, colind, shape, a_nnz] =
    mc::generate_bcsr(m, n, block_height, block_width, nnz);

mc::bcsr_matrix_view view(values.begin(), rowptr.begin(), colind.begin(),
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

```c++
auto [values, rowptr, colind, shape, nnz] =
    mc::generate_bcsr(m, n, block_height, block_width, nnz);

std::experimental::mdspan a(values(), rowptr, colind, block_shape);

for (auto && [{bx, by}, block] : mc::blocks(a)) {
  auto values = std::ranges::views::values(block);
  fmt::print("A {} x {} block at {}, {} containing values {}\n",
                   block_height, block_width, bx, by, values);
}
```

## <a name='Application'></a>Application 

### <a name='SpMV'></a>SpMV

The processing flow of SpMV $c=Ab$ is designed as follows:
+ Iterate over each block in a sparse matrix. The block iterator is provided by specific interface. The details is transparent to user.
+ Iterate over each element in block and calculate its indices to determine the corresponding indices in $b$ and $c$. Add the resul back to $c$.

```c++

mc::bcsr_matrix_view view(values.begin(), rowptr.begin(),
              colind.begin(), shape, block_height, block_width, nnz);

auto blocks = view.blocks();

std::for_each (blocks.begin(), blocks.end(), [&](Block b) {
  auto x_base = bx * b.shape()[0];
  auto y_base = by * b.shape()[1];
  for (auto i : __ranges::views::iota(I(0), b.shape()[0])) {
    for (auto j : __ranges::views::iota(I(0), b.shape()[1])) {
      auto x_addr = x_base + i;
      auto y_addr = y_base + j;
      C[x_addr] += b[{i, j}] * B[y_addr];
    }
  }
});
```

In this code, `block.shape()[0]` is the block size along the row dimension, and `block.shape()[1]` is the block size along the column dimension.

<img src="fig/block-3.png" width="80%">


### Parallelism 

```c++
/// Initialization
auto [x, shape] = mc::generate_dense(n, 1);
auto [y, shape] = mc::generate_dense(m, 1);
auto [values, rowptr, colind, shape, nnz] =
  mc::generate_bcsr(m, n, block_height, block_width, nnz);

std::experimental::mdspan b(x.data(), shape[0], shape[1]);
std::experimental::mdspan c(y.data(), shape[0], shape[1]);
mc::bcsr_matrix_view A(values.begin(), rowptr.begin(), colind.begin(),
                          shape, block_height, block_width, nnz);
```
