#include "poisson/poisson.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace poisson {

void apply(const Grid& u, Grid& out) {
    const double inv_h2 = 1.0 / (u.h * u.h);
    for (int i = 1; i <= u.n; ++i)
        for (int j = 1; j <= u.n; ++j)
            out(i, j) = (4.0 * u(i, j) - u(i - 1, j) - u(i + 1, j) - u(i, j - 1) - u(i, j + 1)) * inv_h2;
}

double dot(const Grid& a, const Grid& b) {
    double s = 0;
    for (int i = 1; i <= a.n; ++i)
        for (int j = 1; j <= a.n; ++j) s += a(i, j) * b(i, j);
    return s;
}

double norm(const Grid& a) { return std::sqrt(dot(a, a)); }

double residual(const Grid& u, const Grid& f, Grid& r) {
    apply(u, r);
    for (int i = 1; i <= u.n; ++i)
        for (int j = 1; j <= u.n; ++j) r(i, j) = f(i, j) - r(i, j);
    return norm(r) / norm(f);
}

namespace {

// One red-black sweep: update the points with (i + j) % 2 == colour, then the other colour.
void sweep(Grid& u, const Grid& f, double omega, int first_colour) {
    const double h2 = u.h * u.h;
    for (int c = 0; c < 2; ++c) {
        const int colour = (first_colour + c) % 2;
        for (int i = 1; i <= u.n; ++i)
            for (int j = 1 + (i + 1 + colour) % 2; j <= u.n; j += 2) {
                const double gs = 0.25 * (h2 * f(i, j) + u(i - 1, j) + u(i + 1, j) + u(i, j - 1) + u(i, j + 1));
                u(i, j) += omega * (gs - u(i, j));
            }
    }
}

template <class Precondition>
Stats cg_impl(Grid& u, const Grid& f, double tol, int max_iter, Precondition&& precondition) {
    Stats st;
    Grid r(u.n), z(u.n), p(u.n), ap(u.n);
    st.rel_residual = residual(u, f, r);
    const double fnorm = norm(f);
    precondition(r, z);
    p.v = z.v;
    double rz = dot(r, z);
    while (st.iterations < max_iter && st.rel_residual > tol) {
        apply(p, ap);
        const double alpha = rz / dot(p, ap);
        for (int i = 1; i <= u.n; ++i)
            for (int j = 1; j <= u.n; ++j) {
                u(i, j) += alpha * p(i, j);
                r(i, j) -= alpha * ap(i, j);
            }
        ++st.iterations;
        st.rel_residual = norm(r) / fnorm;
        st.history.push_back(st.rel_residual);
        precondition(r, z);
        const double rz_next = dot(r, z);
        const double beta = rz_next / rz;
        rz = rz_next;
        for (int i = 1; i <= u.n; ++i)
            for (int j = 1; j <= u.n; ++j) p(i, j) = z(i, j) + beta * p(i, j);
    }
    return st;
}

// Full-weighting restriction of a fine residual onto the grid with half the resolution.
void restrict_full_weighting(const Grid& fine, Grid& coarse) {
    for (int I = 1; I <= coarse.n; ++I)
        for (int J = 1; J <= coarse.n; ++J) {
            const int i = 2 * I, j = 2 * J;
            coarse(I, J) = (4.0 * fine(i, j) + 2.0 * (fine(i - 1, j) + fine(i + 1, j) + fine(i, j - 1) + fine(i, j + 1)) +
                            fine(i - 1, j - 1) + fine(i - 1, j + 1) + fine(i + 1, j - 1) + fine(i + 1, j + 1)) /
                           16.0;
        }
}

// Bilinear interpolation of a coarse correction, added onto the fine grid.
void prolong_and_add(const Grid& coarse, Grid& fine) {
    for (int i = 1; i <= fine.n; ++i)
        for (int j = 1; j <= fine.n; ++j) {
            const int I0 = i / 2, J0 = j / 2, I1 = (i + 1) / 2, J1 = (j + 1) / 2;
            fine(i, j) += 0.25 * (coarse(I0, J0) + coarse(I0, J1) + coarse(I1, J0) + coarse(I1, J1));
        }
}

void v_cycle(Grid& u, const Grid& f) {
    if (u.n <= 3) {
        for (int k = 0; k < 50; ++k) sweep(u, f, 1.0, 0);  // tiny grid: smooth to convergence
        return;
    }
    for (int k = 0; k < 2; ++k) sweep(u, f, 1.0, 0);  // pre-smooth (red, then black)
    Grid r(u.n);
    residual(u, f, r);
    Grid rc((u.n - 1) / 2), ec((u.n - 1) / 2);
    restrict_full_weighting(r, rc);
    v_cycle(ec, rc);
    prolong_and_add(ec, u);
    for (int k = 0; k < 2; ++k) sweep(u, f, 1.0, 1);  // post-smooth (black, then red)
}

}  // namespace

Stats jacobi(Grid& u, const Grid& f, double tol, int max_iter) {
    Stats st;
    Grid next = u, r(u.n);
    const double h2 = u.h * u.h;
    st.rel_residual = residual(u, f, r);
    while (st.iterations < max_iter && st.rel_residual > tol) {
        for (int i = 1; i <= u.n; ++i)
            for (int j = 1; j <= u.n; ++j)
                next(i, j) = 0.25 * (h2 * f(i, j) + u(i - 1, j) + u(i + 1, j) + u(i, j - 1) + u(i, j + 1));
        u.v.swap(next.v);
        ++st.iterations;
        st.rel_residual = residual(u, f, r);
        st.history.push_back(st.rel_residual);
    }
    return st;
}

double optimal_sor_omega(int n) { return 2.0 / (1.0 + std::sin(std::numbers::pi / (n + 1))); }

Stats sor(Grid& u, const Grid& f, double tol, int max_iter, double omega) {
    Stats st;
    Grid r(u.n);
    st.rel_residual = residual(u, f, r);
    while (st.iterations < max_iter && st.rel_residual > tol) {
        sweep(u, f, omega, 0);
        ++st.iterations;
        st.rel_residual = residual(u, f, r);
        st.history.push_back(st.rel_residual);
    }
    return st;
}

Stats conjugate_gradient(Grid& u, const Grid& f, double tol, int max_iter) {
    return cg_impl(u, f, tol, max_iter, [](const Grid& r, Grid& z) { z.v = r.v; });
}

Stats pcg_ic0(Grid& u, const Grid& f, double tol, int max_iter) {
    const int n = u.n;
    const double h2 = u.h * u.h;
    // For h^2 A (diagonal 4, neighbours -1), IC(0) keeps the sparsity of A, so the
    // factor M = (D + L) D^-1 (D + L^T) is fixed by the pivots
    //   d_ij = 4 - 1/d_{i-1,j} - 1/d_{i,j-1}.
    Grid d(n);
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
            d(i, j) = 4.0 - (i > 1 ? 1.0 / d(i - 1, j) : 0.0) - (j > 1 ? 1.0 / d(i, j - 1) : 0.0);

    Grid y(n);
    auto precondition = [&](const Grid& r, Grid& z) {
        for (int i = 1; i <= n; ++i)  // forward solve (D + L) y = h^2 r
            for (int j = 1; j <= n; ++j) y(i, j) = (h2 * r(i, j) + y(i - 1, j) + y(i, j - 1)) / d(i, j);
        for (int i = n; i >= 1; --i)  // backward solve (D + L^T) z = D y
            for (int j = n; j >= 1; --j) z(i, j) = y(i, j) + (z(i + 1, j) + z(i, j + 1)) / d(i, j);
    };
    return cg_impl(u, f, tol, max_iter, precondition);
}

Stats multigrid(Grid& u, const Grid& f, double tol, int max_cycles) {
    if (((u.n + 1) & u.n) != 0) throw std::invalid_argument("multigrid needs n = 2^k - 1");
    Stats st;
    Grid r(u.n);
    st.rel_residual = residual(u, f, r);
    while (st.iterations < max_cycles && st.rel_residual > tol) {
        v_cycle(u, f);
        ++st.iterations;
        st.rel_residual = residual(u, f, r);
        st.history.push_back(st.rel_residual);
    }
    return st;
}

}  // namespace poisson
