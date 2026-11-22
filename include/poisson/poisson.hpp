#pragma once

#include <cstddef>
#include <vector>

namespace poisson {

// Unknowns on an n x n grid of interior points of the unit square, spacing
// h = 1/(n+1), stored with a one-cell ring of zeros for the Dirichlet boundary.
struct Grid {
    int n;
    double h;
    std::vector<double> v;

    explicit Grid(int n_interior) : n(n_interior), h(1.0 / (n_interior + 1)), v((n + 2) * (n + 2), 0.0) {}
    double& operator()(int i, int j) { return v[static_cast<std::size_t>(i) * (n + 2) + j]; }
    double operator()(int i, int j) const { return v[static_cast<std::size_t>(i) * (n + 2) + j]; }
};

// The 5-point discrete Laplacian (sign chosen so A is symmetric positive definite):
//   (A u)_ij = (4 u_ij - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1}) / h^2
void apply(const Grid& u, Grid& out);
double dot(const Grid& a, const Grid& b);  // over interior points
double norm(const Grid& a);
// r = f - A u; returns ||r|| / ||f||
double residual(const Grid& u, const Grid& f, Grid& r);

struct Stats {
    int iterations = 0;
    double rel_residual = 0;
    std::vector<double> history;  // relative residual after each iteration
};

Stats jacobi(Grid& u, const Grid& f, double tol, int max_iter);
// Successive over-relaxation in red-black order; omega = 1 is Gauss-Seidel.
Stats sor(Grid& u, const Grid& f, double tol, int max_iter, double omega);
double optimal_sor_omega(int n);  // 2 / (1 + sin(pi h))
Stats conjugate_gradient(Grid& u, const Grid& f, double tol, int max_iter);
// CG preconditioned with the incomplete Cholesky factorisation IC(0).
Stats pcg_ic0(Grid& u, const Grid& f, double tol, int max_iter);
// Geometric multigrid V(2,2) cycles; n must be 2^k - 1.
Stats multigrid(Grid& u, const Grid& f, double tol, int max_cycles);

}  // namespace poisson
