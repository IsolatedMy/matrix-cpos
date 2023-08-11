#include <mc/mc.hpp>

#include <vector>
#include <numeric>

#include <fmt/core.h>
#include <fmt/ranges.h>

template <std::ranges::forward_range R>
void range(R&&) {
  fmt::print("forward_range\n");
}

template <std::ranges::bidirectional_range R>
void range(R&&) {
  fmt::print("bidirectional_range\n");
}

template <std::ranges::random_access_range R>
void range(R&&) {
  fmt::print("random_access_range\n");
}

template <std::ranges::contiguous_range R>
void range(R&&) {
  fmt::print("contiguous_range\n");
}

int main(int argc, char** argv) {
  std::size_t m = 10;
  std::size_t n = 10;
  std::vector<int> v(m*n);

  std::iota(v.begin(), v.end(), 0);

  mc::dense_matrix_view view(v.begin(), {m, n});

  fmt::print("{}\n", v);

  for (std::size_t i = 0; i < view.shape()[0]; i++) {
    for (std::size_t j = 0; j < view.shape()[1]; j++) {
      fmt::print("{:2d}", view[{i, j}]);
      if (j != view.shape()[1]-1) {
        fmt::print(", ");
      }
    }
    fmt::print("\n");
  }

  for (std::size_t i = 0; i < view.shape()[0]; i++) {
    auto row = view.row(i);
    fmt::print("{}\n", row);
    auto row_values = std::ranges::views::values(row);
    fmt::print("{}\n", row_values);
  }

  for (std::size_t j = 0; j < view.shape()[1]; j++) {
    auto col = view.column(j);
    fmt::print("{}\n", col);
    auto col_values = std::ranges::views::values(col);
    fmt::print("{}\n", col_values);
  }

  return 0;
}
