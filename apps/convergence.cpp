// Compare solvers on the 2D Poisson problem with u = sin(pi x) sin(pi y).
// usage: convergence [output_dir]
// Prints iteration/time tables and writes history.csv (n = 255 residual histories)
// and order.csv (discretisation error vs h).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numbers>
#include <string>
#include <vector>

#include "poisson/poisson.hpp"

using namespace poisson;
using std::numbers::pi;
using Clock = std::chrono::steady_clock;

namespace {
// Solver comparison uses u = x(1-x) y(1-y), which the 5-point stencil reproduces exactly and
// which is not an eigenvector of the discrete Laplacian (sin(pi x) sin(pi y) is, and CG would
// solve it in one step).
Grid rhs_poly(int n) {
    Grid f(n);
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j) {
            const double x = i * f.h, y = j * f.h;
            f(i, j) = 2.0 * (x * (1 - x) + y * (1 - y));
        }
    return f;
}
// Order of accuracy uses u = sin(pi x) sin(pi y), which has a non-zero truncation error.
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
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    const double tol = 1e-8;
    using Solver = std::function<Stats(Grid&, const Grid&)>;
    const std::vector<std::pair<std::string, Solver>> solvers = {
        {"Jacobi", [&](Grid& u, const Grid& f) { return jacobi(u, f, tol, 2'000'000); }},
        {"SOR (optimal omega)", [&](Grid& u, const Grid& f) { return sor(u, f, tol, 1'000'000, optimal_sor_omega(u.n)); }},
        {"CG", [&](Grid& u, const Grid& f) { return conjugate_gradient(u, f, tol, 100'000); }},
        {"PCG + IC(0)", [&](Grid& u, const Grid& f) { return pcg_ic0(u, f, tol, 100'000); }},
        {"Multigrid V(2,2)", [&](Grid& u, const Grid& f) { return multigrid(u, f, tol, 1000); }},
    };

    std::printf("relative residual 1e-8: iterations (time)\n%-6s", "n");
    for (const auto& [name, _] : solvers) std::printf(" | %-22s", name.c_str());
    std::printf("\n");
    std::FILE* hist = std::fopen((dir + "/history.csv").c_str(), "w");
    std::FILE* order = std::fopen((dir + "/order.csv").c_str(), "w");
    std::FILE* table = std::fopen((dir + "/iterations.csv").c_str(), "w");
    if (!hist || !order || !table) {
        std::fprintf(stderr, "cannot write to %s (does the directory exist?)\n", dir.c_str());
        return 1;
    }
    std::fprintf(hist, "solver,iteration,rel_residual\n");
    std::fprintf(table, "solver,n,iterations,ms\n");
    for (int n : {31, 63, 127, 255, 511}) {
        std::printf("%-6d", n);
        const Grid f = rhs_poly(n);
        for (const auto& [name, solve] : solvers) {
            if (name == "Jacobi" && n > 127) {
                std::printf(" | %-22s", "(too slow)");
                continue;
            }
            Grid u(n);
            const auto t0 = Clock::now();
            const Stats st = solve(u, f);
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            char cell[64];
            std::snprintf(cell, sizeof cell, "%d (%.1f ms)", st.iterations, ms);
            std::printf(" | %-22s", cell);
            std::fprintf(table, "\"%s\",%d,%d,%.3f\n", name.c_str(), n, st.iterations, ms);
            if (n == 255)
                for (std::size_t k = 0; k < st.history.size(); ++k)
                    std::fprintf(hist, "\"%s\",%zu,%.6e\n", name.c_str(), k + 1, st.history[k]);
        }
        std::printf("\n");
    }
    std::fclose(hist);
    std::fclose(table);

    std::fprintf(order, "n,h,max_error\n");
    std::printf("\n%-6s %-12s %-12s %s\n", "n", "h", "max error", "observed order");
    double prev = 0;
    for (int n : {7, 15, 31, 63, 127, 255, 511}) {
        Grid u(n);
        multigrid(u, rhs(n), 1e-13, 200);
        const double e = max_error(u);
        std::fprintf(order, "%d,%.8e,%.8e\n", n, u.h, e);
        if (prev > 0)
            std::printf("%-6d %-12.3e %-12.3e %.3f\n", n, u.h, e, std::log2(prev / e));
        else
            std::printf("%-6d %-12.3e %-12.3e\n", n, u.h, e);
        prev = e;
    }
    std::fclose(order);
    return 0;
}
