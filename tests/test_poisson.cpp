#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "poisson/poisson.hpp"

namespace {
int g_failures = 0;
std::vector<std::pair<std::string, std::function<void()>>>& registry() {
    static std::vector<std::pair<std::string, std::function<void()>>> tests;
    return tests;
}
struct Register {
    Register(const char* name, std::function<void()> fn) { registry().emplace_back(name, std::move(fn)); }
};
}  // namespace

#define TEST(name)                                      \
    static void name();                                 \
    static const Register register_##name(#name, name); \
    static void name()

#define CHECK(...)                                                                               \
    do {                                                                                         \
        if (!(__VA_ARGS__)) {                                                                    \
            std::fprintf(stderr, "  %s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #__VA_ARGS__); \
            ++g_failures;                                                                        \
        }                                                                                        \
    } while (0)

using namespace poisson;
using std::numbers::pi;

namespace {
// Two manufactured problems:
//  * u = sin(pi x) sin(pi y): smooth, with a non-zero truncation error, so it measures the
//    discretisation order. It is also an exact eigenvector of the discrete Laplacian, so CG
//    solves it in one step; that makes it useless for comparing solvers.
//  * u = x(1-x) y(1-y), f = 2 [x(1-x) + y(1-y)]: the 5-point stencil is exact for it, so any
//    remaining error is purely algebraic. This is the one to compare solvers on.
Grid rhs_poly(int n) {
    Grid f(n);
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j) {
            const double x = i * f.h, y = j * f.h;
            f(i, j) = 2.0 * (x * (1 - x) + y * (1 - y));
        }
    return f;
}
double max_error_poly(const Grid& u) {
    double e = 0;
    for (int i = 1; i <= u.n; ++i)
        for (int j = 1; j <= u.n; ++j) {
            const double x = i * u.h, y = j * u.h;
            e = std::max(e, std::abs(u(i, j) - x * (1 - x) * y * (1 - y)));
        }
    return e;
}
Grid rhs(int n) {
    Grid f(n);
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j) f(i, j) = 2 * pi * pi * std::sin(pi * i * f.h) * std::sin(pi * j * f.h);
    return f;
}
double max_error(const Grid& u) {
    double e = 0;
    for (int i = 1; i <= u.n; ++i)
        for (int j = 1; j <= u.n; ++j) e = std::max(e, std::abs(u(i, j) - std::sin(pi * i * u.h) * std::sin(pi * j * u.h)));
    return e;
}
double max_diff(const Grid& a, const Grid& b) {
    double e = 0;
    for (std::size_t k = 0; k < a.v.size(); ++k) e = std::max(e, std::abs(a.v[k] - b.v[k]));
    return e;
}
}  // namespace

TEST(laplacian_is_symmetric) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> u01(-1, 1);
    Grid a(17), b(17), aa(17), ab(17);
    for (int i = 1; i <= 17; ++i)
        for (int j = 1; j <= 17; ++j) a(i, j) = u01(rng), b(i, j) = u01(rng);
    apply(a, aa);
    apply(b, ab);
    CHECK(std::abs(dot(aa, b) - dot(a, ab)) < 1e-12 * std::abs(dot(aa, b)));
}

TEST(discretisation_error_is_second_order) {
    std::vector<double> err;
    for (int n : {15, 31, 63, 127}) {
        Grid u(n);
        multigrid(u, rhs(n), 1e-13, 100);
        err.push_back(max_error(u));
    }
    for (std::size_t k = 0; k + 1 < err.size(); ++k) {
        const double order = std::log2(err[k] / err[k + 1]);
        CHECK(order > 1.9 && order < 2.1);
    }
}

TEST(all_solvers_reach_the_exact_discrete_solution) {
    const int n = 31;
    const Grid f = rhs_poly(n);
    Grid uj(n), us(n), uc(n), up(n), um(n);
    CHECK(jacobi(uj, f, 1e-11, 100000).rel_residual <= 1e-11);
    CHECK(sor(us, f, 1e-11, 100000, optimal_sor_omega(n)).rel_residual <= 1e-11);
    CHECK(conjugate_gradient(uc, f, 1e-11, 10000).rel_residual <= 1e-11);
    CHECK(pcg_ic0(up, f, 1e-11, 10000).rel_residual <= 1e-11);
    CHECK(multigrid(um, f, 1e-11, 100).rel_residual <= 1e-11);
    for (const Grid* g : {&uj, &us, &uc, &up, &um}) CHECK(max_error_poly(*g) < 1e-9);
    CHECK(max_diff(uc, um) < 1e-9);
}

TEST(cg_converges_within_the_number_of_unknowns) {
    const int n = 9;  // 81 unknowns: exact arithmetic would need at most 81 steps
    Grid u(n);
    const Stats st = conjugate_gradient(u, rhs_poly(n), 1e-10, 81);
    CHECK(st.rel_residual <= 1e-10);
    CHECK(st.iterations <= 81);
}

TEST(ic0_preconditioning_cuts_cg_iterations) {
    const int n = 127;
    Grid a(n), b(n);
    const int plain = conjugate_gradient(a, rhs_poly(n), 1e-8, 10000).iterations;
    const int preconditioned = pcg_ic0(b, rhs_poly(n), 1e-8, 10000).iterations;
    CHECK(preconditioned < 0.6 * plain);
}

TEST(optimal_sor_is_far_faster_than_gauss_seidel) {
    const int n = 63;
    Grid a(n), b(n);
    const int gs = sor(a, rhs_poly(n), 1e-8, 100000, 1.0).iterations;
    const int opt = sor(b, rhs_poly(n), 1e-8, 100000, optimal_sor_omega(n)).iterations;
    CHECK(opt < 0.2 * gs);
}

TEST(multigrid_cycle_count_does_not_grow_with_grid_size) {
    std::vector<int> cycles;
    for (int n : {31, 63, 127, 255}) {
        Grid u(n);
        cycles.push_back(multigrid(u, rhs_poly(n), 1e-10, 100).iterations);
    }
    const auto [lo, hi] = std::minmax_element(cycles.begin(), cycles.end());
    CHECK(*hi - *lo <= 2);
    CHECK(*hi <= 15);
}

int main() {
    for (const auto& [name, fn] : registry()) {
        const int before = g_failures;
        fn();
        std::printf("%s %s\n", g_failures == before ? "PASS" : "FAIL", name.c_str());
    }
    std::printf("%zu tests, %d failed checks\n", registry().size(), g_failures);
    return g_failures == 0 ? 0 : 1;
}
