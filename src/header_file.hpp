#pragma once

#include <armadillo>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <math.h>

double calc_error(const arma::mat &exact, const arma::mat &value)
{
    double total = 0;
    for (arma::uword i = 0; i < exact.n_cols; i++) {
        for (int j = 0; j < 3; j++) {
            total += pow(value(j, i) - exact(j, i), 2);
        }
    }
    return sqrt(total);
}
double norm2(const arma::mat &value)
{
    double total = 0;
    for (arma::uword i = 0; i < value.n_cols; i++) {
        for (int j = 0; j < value.n_rows; j++) {
            total += pow(value(j, i), 2);
        }
    }
    return sqrt(total);
}

int factorial(int n)
{
    int product = 1;
    for (int i = 1; i <= n; i++) {
        product *= i;
    }
    return product;
}
int double_factorial(int n)
{
    int product = 1;
    for (int i = n; i > 0; i -= 2) {
        product *= i;
    }
    return product;
}
double binomial(int m, int n)
{
    return factorial(m) / (factorial(n) * factorial(m - n));
}

class Quadrature
{
  public:
    int refinement;
    virtual double refine() = 0;
    double operator()(const double tol, const int MAXSTEPS = 40)
    {
        double oldvalue;
        for (int i = 0; i < MAXSTEPS; i++) {
            double value = refine();
            if ((i > 0) && ((fabs(value - oldvalue) < tol) ||
                            (value == 0 && oldvalue == 0))) {
                return value;
            }
            oldvalue = value;
        }
        throw("Quadrature did not converge, too many steps.");
    }
};

template <class T> class Trapezoid : Quadrature
{
  private:
    double lower, upper, value;

  public:
    T &f;
    Trapezoid() {}
    Trapezoid(T &F, const double LOWER, const double UPPER)
        : f(F), lower(LOWER), upper(UPPER)
    {
        refinement = 0;
    }
    double refine()
    {
        if (refinement++ == 0) {
            value = (upper - lower) * (f(lower) + f(upper)) / 2.0;
        } else {
            int nsteps = 1 << (refinement - 2);
            double del = (upper - lower) / nsteps;
            double x = lower + del / 2.0;
            double sum = 0.0;
            for (int i = 0; i < nsteps; i++, x += del) {
                sum += f(x);
            }
            value = (value + (upper - lower) * sum / nsteps) / 2.0;
        }
        return value;
    }
    double operator()(const double tol, const int MAXSTEPS = 20)
    {
        return Quadrature::operator()(tol, MAXSTEPS);
    }
};

class Function
{
  public:
    virtual double operator()(double x) = 0;
};

class DoubleGaussian : Function
{
  private:
    double XA, XB, alpha, beta;
    int lA, lB;

  public:
    DoubleGaussian(const double XA_in, const int lA, const double alpha,
                   const double XB, const int lB, const double beta)
        : XA(XA), lA(lA), alpha(alpha), XB(XB), lB(lB), beta(beta)
    {
    }
    double operator()(double x)
    {
        double dxA = x - XA;
        double dxB = x - XB;
        return pow(dxA, lA) * pow(dxB, lB) *
               exp(-alpha * dxA * dxA - beta * dxB * dxB);
    }
};

int n_functions(const int l) { return (l + 1) * (l + 2) / 2; }

arma::umat lmat(const int l)
{
    arma::uword count = n_functions(l);
    arma::umat result(3, count);
    for (arma::uword i = 0; i <= l; i++) {
        for (arma::uword j = 0; i + j <= l; j++) {
            count--;
            result(0, count) = i;
            result(1, count) = j;
            result(2, count) = l - i - j;
        }
    }
    return result;
}
class GAO
{
  public:
    arma::vec X;
    double alpha;
    arma::uvec l;
    GAO()
    {
        X = arma::vec(3);
        l = arma::uvec(3);
    }
    GAO(const arma::vec &X, const double alpha, const arma::uvec l)
        : X(X), alpha(alpha), l(l)
    {
    }
    friend std::istream &operator>>(std::istream &in, GAO &basis);
    friend std::ostream &operator<<(std::ostream &out, const GAO &basis);
};

// std::istream & operator >> (std::istream &in, GAO& basis) {
//     in >> basis.X(0) >> basis.X(1) >> basis.X(2) >> basis.alpha >> basis.l;
//     return in;
// }
// std::ostream & operator << (std::ostream &out, const GAO& basis) {
//     out << std::setprecision(2) << std::fixed;
//     out << "This shell info: R( " << basis.X(0) << ", " << basis.X(1) << ", "
//     << basis.X(2) << "), with angular momentum: " << shell.l << ",
//     coefficient: " << shell.alpha; out << std::defaultfloat; return out;
// }

double overlap(const GAO &A, const GAO &B, const arma::uword dir)
{
    double alpha = A.alpha;
    double beta = B.alpha;
    arma::vec RP = (alpha * A.X + beta * B.X) / (alpha + beta);
    arma::uword lA = A.l(dir);
    arma::uword lB = B.l(dir);
    double sum = 0.0;
    double XA = A.X(dir);
    double XB = B.X(dir);
    double XP = RP(dir);
    double prefactor =
        exp(-alpha * beta * (XA - XB) * (XA - XB) / (alpha + beta)) *
        sqrt(M_PI / (alpha + beta));
    for (arma::uword i = 0; i <= lA; i++) {
        for (arma::uword j = 0; j <= lB; j++) {
            if ((i + j) % 2 == 0) {
                sum += binomial(lA, i) * binomial(lB, j) *
                       double_factorial(i + j - 1) * pow(XP - XA, lA - i) *
                       pow(XP - XB, lB - j) /
                       pow(2.0 * (alpha + beta), (i + j) / 2.0);
            }
        }
    }
    return prefactor * sum;
}

double overlap(const GAO &A, const GAO &B)
{
    double result = 1.0;
    for (arma::uword dir = 0; dir < 3; dir++) {
        result *= overlap(A, B, dir);
    }
    return result;
}

arma::vec overlap_grad(const GAO &A, const GAO &B)
{
    double alpha = A.alpha;
    double beta = B.alpha;
    arma::vec RP = (alpha * A.X + beta * B.X) / (alpha + beta);
    arma::vec result = arma::ones(3);
    if (norm2(A.X - B.X) < std::numeric_limits<double>::epsilon()) {
        return arma::zeros(3);
    }
    for (arma::uword dir = 0; dir < 3; dir++) {
        int lA = A.l(dir);
        int lB = B.l(dir);
        double XA = A.X(dir);
        double XB = B.X(dir);
        double XP = RP(dir);
        double prefactor =
            exp(-alpha * beta * (XA - XB) * (XA - XB) / (alpha + beta)) *
            sqrt(M_PI / (alpha + beta));
        double sum = 0.0;
        for (arma::uword i = 0; i <= lA + 1; i++) {
            for (int j = 0; j <= lB; j++) {
                if ((i + j) % 2 == 0) {
                    if (i <= lA - 1) {
                        sum -= lA * binomial(lA - 1, i) * binomial(lB, j) *
                               double_factorial(i + j - 1) *
                               pow(XP - XA, lA - 1 - i) * pow(XP - XB, lB - j) /
                               pow(2.0 * (alpha + beta), (i + j) / 2.0);
                    }
                    sum += 2 * alpha * binomial(lA + 1, i) * binomial(lB, j) *
                           double_factorial(i + j - 1) *
                           pow(XP - XA, lA + 1 - i) * pow(XP - XB, lB - j) /
                           pow(2.0 * (alpha + beta), (i + j) / 2.0);
                }
            }
        }
        result(dir) *= prefactor * sum;
        for (arma::uword other_dir = 0; other_dir < 3; other_dir++) {
            if (other_dir != dir) {
                result(dir) *= overlap(A, B, other_dir);
            }
        }
    }
    return -result;
}

double eri(const GAO &A, const GAO &Ap, const GAO &B, const GAO &Bp)
{
    double sigmaA = 1.0 / (A.alpha + Ap.alpha);
    double sigmaB = 1.0 / (B.alpha + Bp.alpha);
    double V2 = 1.0 / (sigmaA + sigmaB);
    double U = pow(M_PI * sigmaA * M_PI * sigmaB, 1.5);
    double T = V2 * arma::dot(A.X - B.X, A.X - B.X);
    if (T == 0.0) {
        return 27.211324570273 * U * sqrt(2.0 * V2 * 2.0 / M_PI);
    } else {
        return 27.211324570273 * U / arma::norm(A.X - B.X) * std::erf(sqrt(T));
    }
}

arma::vec eri_grad(const GAO &A, const GAO &Ap, const GAO &B, const GAO &Bp)
{
    double sigmaA = 1.0 / (A.alpha + Ap.alpha);
    double sigmaB = 1.0 / (B.alpha + Bp.alpha);
    double V2 = 1.0 / (sigmaA + sigmaB);
    double U = pow(M_PI * sigmaA * M_PI * sigmaB, 1.5);
    double T = V2 * arma::dot(A.X - B.X, A.X - B.X);
    if (T == 0.0) {
        return arma::zeros(3);
    } else {
        return 27.211324570273 * U * (A.X - B.X) /
               pow(arma::norm(A.X - B.X), 2) *
               (-std::erf(sqrt(T)) / arma::norm(A.X - B.X) +
                2.0 * sqrt(V2 / M_PI) * exp(-T));
    }
}

class CGAO
{
  public:
    arma::vec X;
    arma::uvec l;
    arma::vec d;
    arma::vec N;
    double h;
    double IA;
    int beta;
    std::vector<GAO> basis;
    CGAO()
    {
        X = arma::vec(3);
        l = arma::uvec(3);
        d = arma::vec(3);
        N = arma::vec(3);
        basis = std::vector<GAO>();
    }
    CGAO(const arma::vec X, const arma::uvec l, const int atomicNumber)
        : X(X), l(l)
    {
        arma::vec alpha;
        if (atomicNumber == 1 && arma::sum(l) == 0) {
            alpha = {3.42525091, 0.62391373, 0.16885540};
            d = {0.15432897, 0.53532814, 0.44463454};
            h = -13.6;
            IA = 2.0 * 7.176;
            beta = -9;
        } else if (atomicNumber == 6 && arma::sum(l) == 0) {
            alpha = {2.94124940, 0.68348310, 0.22228990};
            d = {-0.09996723, 0.39951283, 0.70011547};
            h = -21.4;
            IA = 2.0 * 14.051;
            beta = -21;
        } else if (atomicNumber == 6 && arma::sum(l) == 1) {
            alpha = {2.94124940, 0.68348310, 0.22228990};
            d = {0.15591627, 0.60768372, 0.39195739};
            h = -11.4;
            IA = 2.0 * 5.572;
            beta = -21;
        } else if (atomicNumber == 7 && arma::sum(l) == 0) {
            alpha = {3.78045590, 0.87849660, 0.28571440};
            d = {-0.09996723, 0.39951283, 0.70011547};
            h = -21.4;
            IA = 2.0 * 19.316;
            beta = -25;
        } else if (atomicNumber == 7 && arma::sum(l) == 1) {
            alpha = {3.78045590, 0.87849660, 0.28571440};
            d = {0.15591627, 0.60768372, 0.39195739};
            h = -11.4;
            IA = 2.0 * 7.275;
            beta = -25;
        } else if (atomicNumber == 8 && arma::sum(l) == 0) {
            alpha = {5.03315130, 1.16959610, 0.38038900};
            d = {-0.09996723, 0.39951283, 0.70011547};
            h = -21.4;
            IA = 2.0 * 25.390;
            beta = -31;
        } else if (atomicNumber == 8 && arma::sum(l) == 1) {
            alpha = {5.03315130, 1.16959610, 0.38038900};
            d = {0.15591627, 0.60768372, 0.39195739};
            h = -11.4;
            IA = 2.0 * 9.111;
            beta = -31;
        } else if (atomicNumber == 9 && arma::sum(l) == 0) {
            alpha = {6.46480320, 1.50228120, 0.48858850};
            d = {-0.09996723, 0.39951283, 0.70011547};
            h = -21.4;
            IA = 2.0 * 32.272;
            beta = -39;
        } else if (atomicNumber == 9 && arma::sum(l) == 1) {
            alpha = {6.46480320, 1.50228120, 0.48858850};
            d = {0.15591627, 0.60768372, 0.39195739};
            h = -11.4;
            IA = 2.0 * 11.080;
            beta = -39;
        } else {
            throw std::invalid_argument("Element or orbital not supported!");
        }
        N = arma::vec(alpha.n_elem);
        for (arma::uword i = 0; i < alpha.n_elem; i++) {
            GAO element = GAO(X, alpha(i), l);
            basis.push_back(element);
            N(i) = pow(overlap(element, element), -0.5);
        }
    }
};

double overlap(const CGAO &A, const CGAO &B)
{
    double result = 0.0;
    for (arma::uword i = 0; i < A.basis.size(); i++) {
        for (arma::uword j = 0; j < B.basis.size(); j++) {
            result += A.d(i) * A.N(i) * B.d(j) * B.N(j) *
                      overlap(A.basis[i], B.basis[j]);
        }
    }
    return result;
}

arma::vec overlap_grad(const CGAO &A, const CGAO &B)
{
    arma::vec result = arma::zeros(3);
    for (arma::uword i = 0; i < A.basis.size(); i++) {
        for (arma::uword j = 0; j < B.basis.size(); j++) {
            result += A.d(i) * A.N(i) * B.d(j) * B.N(j) *
                      overlap_grad(A.basis[i], B.basis[j]);
        }
    }
    return result;
}

double eri(const CGAO &A, const CGAO &B)
{
    double result = 0.0;
    arma::vec Adp = A.d % A.N;
    arma::vec Bdp = B.d % B.N;
    for (arma::uword k = 0; k < 3; k++) {
        for (arma::uword kp = 0; kp < 3; kp++) {
            for (arma::uword l = 0; l < 3; l++) {
                for (arma::uword lp = 0; lp < 3; lp++) {
                    result +=
                        Adp(k) * Adp(kp) * Bdp(l) * Bdp(lp) *
                        eri(A.basis[k], A.basis[kp], B.basis[l], B.basis[lp]);
                }
            }
        }
    }
    return result;
}

arma::vec eri_grad(const CGAO &A, const CGAO &B)
{
    arma::vec result = arma::zeros(3);
    arma::vec Adp = A.d % A.N;
    arma::vec Bdp = B.d % B.N;
    for (arma::uword k = 0; k < 3; k++) {
        for (arma::uword kp = 0; kp < 3; kp++) {
            for (arma::uword l = 0; l < 3; l++) {
                for (arma::uword lp = 0; lp < 3; lp++) {
                    result += Adp(k) * Adp(kp) * Bdp(l) * Bdp(lp) *
                              eri_grad(A.basis[k], A.basis[kp], B.basis[l],
                                       B.basis[lp]);
                }
            }
        }
    }
    return result;
}

class Atom
{
  public:
    int E;
    int Z;
    arma::vec r;
    std::vector<CGAO> orbitals;
    bool use_orbitals;
    Atom(int E, const arma::vec &r, const bool USE_ORBITALS = true)
        : E(E), r(r), use_orbitals(USE_ORBITALS)
    {
        if (use_orbitals) {
            build_orbitals();
        }
    }

    void build_orbitals()
    {
        orbitals = std::vector<CGAO>();
        if (E <= 10) {
            arma::umat ls = lmat(0);
            for (arma::uword i = 0; i < ls.n_cols; i++) {
                orbitals.push_back(CGAO(r, ls.col(i), E));
            }
            if (E <= 2) {
                Z = E;
            } else if (E <= 4) {
                for (arma::uword i = 0; i < ls.n_cols; i++) {
                    orbitals.push_back(CGAO(r, ls.col(i), E));
                }
                Z = E - 2;
            } else {
                ls = lmat(1);
                for (arma::uword i = 0; i < ls.n_cols; i++) {
                    orbitals.push_back(CGAO(r, ls.col(i), E));
                }
                Z = E - 2;
            }
        } else {
            throw std::invalid_argument("Element not supported!");
        }
    }

    void move(const arma::vec &forces, const double step)
    {
        r += step * forces;
        if (use_orbitals) {
            build_orbitals();
        }
    }
};

std::ostream &operator<<(std::ostream &os, const Atom &atom)
{
    return (os << atom.E << "(" << atom.r[0] << ", " << atom.r[1] << ", "
               << atom.r[2] << ")");
}

arma::mat overlap(const Atom &A, const Atom &B)
{
    arma::mat result = arma::zeros(A.orbitals.size(), B.orbitals.size());
    for (arma::uword i = 0; i < A.orbitals.size(); i++) {
        for (arma::uword j = 0; j < B.orbitals.size(); j++) {
            result(i, j) = overlap(A.orbitals[i], B.orbitals[j]);
        }
    }
    return result;
}

arma::cube overlap_grad(const Atom &A, const Atom &B)
{
    arma::cube result = arma::zeros(3, A.orbitals.size(), B.orbitals.size());
    for (arma::uword i = 0; i < A.orbitals.size(); i++) {
        for (arma::uword j = 0; j < B.orbitals.size(); j++) {
            result.slice(j).col(i) = overlap_grad(A.orbitals[i], B.orbitals[j]);
        }
    }
    return result;
}

double eri(const Atom &A, const Atom &B)
{
    // Only sum over valence s orbital centered on atoms A and B
    return eri(A.orbitals[0], B.orbitals[0]);
}

arma::vec eri_grad(const Atom &A, const Atom &B)
{
    return eri_grad(A.orbitals[0], B.orbitals[0]);
}

arma::mat density_matrix(const arma::mat &C)
{
    arma::mat P = arma::zeros(C.n_rows, C.n_rows);
    for (arma::uword i = 0; i < C.n_cols; i++) {
        P += C.col(i) * C.col(i).t();
    }
    return P;
}

struct bracket {
    double a;
    double b;
    double c;
};

bracket bracket_minimum(std::function<double(double)> f, double a, double b,
                        const double GOLDEN_RATIO = 1.618304)
{
    double fa = f(a);
    double fb = f(b);
    if (fb > fa) {
        std::swap(fa, fb);
        std::swap(a, b);
    }
    double c = b + GOLDEN_RATIO * (b - a);
    double fc = f(c);
    double fu;
    while (fb > fc) {
        double r = (b - a) * (fb - fc);
        double q = (b - c) * (fb - fa);
        double u = b - ((b - c) * q - (b - a) * r) / (2.0 * (q - r));
        double ulim = b + 100.0 * (c - b);
        if ((b - u) * (u - c) > 0.0) {
            fu = f(u);
            if (fu < fc) {
                a = b;
                fa = fb;
                b = u;
                fb = fu;
                break;
            } else if (fu > fb) {
                c = u;
                fc = fu;
                break;
            }
            u = c + GOLDEN_RATIO * (c - b);
        } else if ((c - u) * (u - ulim) > 0.0) {
            if (fu < fc) {
                fu = f(u);
                b = c;
                fb = fc;
                c = u;
                fc = fu;
                u = u + GOLDEN_RATIO * (u - c);
            }
        } else {
            u = c + GOLDEN_RATIO * (c - b);
        }
        a = b;
        fa = fb;
        b = c;
        fb = fc;
        c = u;
        fc = f(u);
    }
    bracket results = {a, b, c};
    return results;
}

double line_search(std::ostream &fout, std::function<double(double)> f,
                   double a, double b, const double GOLDEN_RATIO = 1.618304,
                   const double TOL = 3.0e-8, const double h = 1e-4,
                   const double l = 0.3, const bool golden_section = true)
{
    double xmin = 0.0;

    if (golden_section) {
        fout << "Start golden section search" << std::endl;
    }
    int flag = 0;
    bracket results;
    if (golden_section) {
        results = bracket_minimum(f, a, b, GOLDEN_RATIO);
        flag = 2;
    }
    if (flag == 2) {
        // Do golden section search
        double x0 = 0.0;
        double x1 = results.a;
        double x2 = results.b;
        double x3 = results.c;
        if (abs(x3 - x2) < abs(x1 - x0)) {
            x2 += (x3 - x2) * (1.0 - 1.0 / GOLDEN_RATIO);
        } else {
            x1 -= (x1 - x0) * (1.0 - 1.0 / GOLDEN_RATIO);
        }
        double f1 = f(x1);
        double f2 = f(x2);
        while (abs(x3 - x0) > TOL * (abs(x1) + abs(x2))) {
            if (f1 < f2) {
                x3 = x2;
                x2 = x1;
                x1 = x1 / GOLDEN_RATIO + x0 * (1.0 - 1.0 / GOLDEN_RATIO);
                f2 = f1;
                f1 = f(x1);
            } else {
                x0 = x1;
                x1 = x2;
                x2 = x2 / GOLDEN_RATIO + x3 * (1.0 - 1.0 / GOLDEN_RATIO);
                f1 = f2;
                f2 = f(x2);
            }
        }
        if (f1 < f2) {
            xmin = x1;
        } else {
            xmin = x2;
        }
    } else if (f(l) <= f(xmin)) {
        xmin = l;
    }
    return xmin;
}

class Simulation
{
  public:
    std::vector<Atom> atoms;

    Simulation(const std::vector<Atom> atoms) : atoms(atoms) {}

    void move(const arma::mat &forces, double step)
    {
        double total = 0.0;
        for (arma::uword i = 0; i < atoms.size(); i++) {
            for (arma::uword j = 0; j < 3; j++) {
                total += pow(forces(j, i), 2);
            }
        }
        step /= sqrt(total);
        for (arma::uword i = 0; i < atoms.size(); i++) {
            atoms[i].move(forces.col(i), step);
        }
    }

    int basis_count() const
    {
        int a = 0;
        int b = 0;
        for (Atom atom : atoms) {
            if (atom.E == 1) {
                b++;
            } else if (atom.E == 6 || atom.E == 7 || atom.E == 8 ||
                       atom.E == 9) {
                a++;
            }
        }
        return 4 * a + b;
    }

    virtual double energy() const
    {
        throw std::logic_error("Base Simulation not implemented!");
    }
};

std::ostream &operator<<(std::ostream &os, const Simulation &sim)
{
    return (os << sim.atoms);
}
class LennardJones : public Simulation
{
  public:
    double sigma;
    double epsilon;
    LennardJones(const std::vector<Atom> atoms, const double SIGMA = 2.951,
                 const double EPSILON = 5.29)
        : Simulation(atoms), sigma(SIGMA), epsilon(EPSILON)
    {
    }
    double energy() const override
    {
        double energy = 0.0;
        for (arma::uword i = 0; i < atoms.size(); i++) {
            for (arma::uword j = i + 1; j < atoms.size(); j++) {
                double R_ij = arma::norm(atoms[i].r - atoms[j].r);
                energy += epsilon *
                          (pow(sigma / R_ij, 12) - 2 * pow(sigma / R_ij, 6));
            }
        }
        return energy;
    }
    arma::mat lennard_jones_numerical_force(const double h, const bool center)
    {
        arma::mat forces = arma::zeros(3, atoms.size());
        double E = energy();
        for (arma::uword i = 0; i < atoms.size(); i++) {
            for (arma::uword k = 0; k < 3; k++) {
                double energy_left = E;
                double energy_right = E;
                for (arma::uword j = 0; j < atoms.size(); j++) {
                    if (i == j) {
                        continue;
                    }
                    double R_ij = norm(atoms[i].r - atoms[j].r);
                    double energy_contribution =
                        epsilon *
                        (pow(sigma / R_ij, 12) - 2 * pow(sigma / R_ij, 6));
                    energy_left -= energy_contribution;
                    energy_right -= energy_contribution;
                    arma::vec atom_left = atoms[i].r;
                    atom_left[k] -= h;
                    arma::vec atom_right = atoms[i].r;
                    atom_right[k] += h;
                    R_ij = norm(atom_left - atoms[j].r);
                    energy_left += epsilon * (pow(sigma / R_ij, 12) -
                                              2 * pow(sigma / R_ij, 6));
                    R_ij = norm(atom_right - atoms[j].r);
                    energy_right += epsilon * (pow(sigma / R_ij, 12) -
                                               2 * pow(sigma / R_ij, 6));
                }
                if (center) {
                    forces(k, i) = -(energy_right - energy_left) / (2.0 * h);
                } else {
                    forces(k, i) = -(energy_right - E) / h;
                }
            }
        }
        return forces;
    }
    arma::mat lennard_jones_analytic_force()
    {
        arma::mat forces = arma::zeros(3, atoms.size());
        for (arma::uword i = 0; i < atoms.size(); i++) {
            for (arma::uword j = 0; j < atoms.size(); j++) {
                if (i == j) {
                    continue;
                }
                double R_ij = norm(atoms[i].r - atoms[j].r);
                double mag = 12 * epsilon *
                             (pow(sigma / R_ij, 12) - pow(sigma / R_ij, 6)) /
                             (R_ij * R_ij);
                forces.col(i) += mag * (atoms[i].r - atoms[j].r);
            }
        }
        return forces;
    }
};

class ExtendedHuckel : public Simulation
{
  public:
    ExtendedHuckel(const std::vector<Atom> atoms) : Simulation(atoms) {}

    int electron_count() const
    {
        const arma::uword N = basis_count();
        if (N % 2 != 0) {
            throw std::invalid_argument("Odd number of valence electrons!");
        }
        return N / 2;
    }

    arma::mat overlap_matrix() const
    {
        const arma::uword N = basis_count();
        arma::mat S = arma::zeros(0, 0);
        for (Atom A : atoms) {
            arma::mat S_i = arma::zeros(0, 0);
            for (Atom B : atoms) {
                arma::mat S_ij = overlap(A, B);
                S_i = arma::join_rows(S_i, S_ij);
            }
            S = arma::join_cols(S, S_i);
        }
        return S;
    }

    arma::mat hamiltonian(const double K = 1.75) const
    {
        const arma::uword N = basis_count();
        arma::mat S = overlap_matrix();
        arma::mat H = arma::zeros(N, N);

        // Set diagonals to ionization potentials
        arma::uword i = 0;
        for (Atom atom : atoms) {
            for (CGAO orbital : atom.orbitals) {
                H(i, i) = orbital.h;
                i++;
            }
        }

        // Compute off-diagonal elements
        for (arma::uword i = 0; i < N; i++) {
            for (arma::uword j = i + 1; j < N; j++) {
                const double H_ij = 0.5 * K * (H(i, i) + H(j, j)) * S(i, j);
                H(i, j) = H_ij;
                H(j, i) = H_ij;
            }
        }

        return H;
    }

    arma::mat x_matrix() const
    {
        arma::mat S = overlap_matrix();
        return arma::inv(arma::sqrtmat_sympd(S));
    }

    double energy() const override
    {
        arma::mat S = overlap_matrix();
        arma::mat H = hamiltonian();
        arma::mat X = x_matrix();
        arma::mat Hdiag = X.t() * H * X;
        arma::vec epsilon;
        arma::mat V;
        arma::eig_sym(epsilon, V, Hdiag);
        double energy = 0.0;
        for (arma::uword i = 0; i < electron_count(); i++) {
            energy += 2 * epsilon(i);
        }
        return energy;
    }

    arma::mat molecular_orbital_coefficients() const
    {
        arma::mat S = overlap_matrix();
        arma::mat H = hamiltonian();
        arma::mat X = x_matrix();
        arma::mat Hdiag = X.t() * H * X;
        arma::vec epsilon;
        arma::mat V;
        arma::eig_sym(epsilon, V, Hdiag);
        return X * V;
    }

    arma::mat molecular_orbital_overlap() const
    {
        arma::mat S = overlap_matrix();
        arma::mat C = molecular_orbital_coefficients();
        return C.t() * S * C;
    }
};

class CNDO2 : public Simulation
{
  public:
    int n_alpha;
    int n_beta;
    arma::mat Calpha;
    arma::mat Cbeta;
    CNDO2(const std::vector<Atom> atoms, const int N_ALPHA, const int N_BETA)
        : Simulation(atoms), n_alpha(N_ALPHA), n_beta(N_BETA)
    {
        int N = basis_count();
        Calpha = arma::zeros(N, n_alpha);
        Cbeta = arma::zeros(N, n_beta);
    }

    void converge(const double tol = std::numeric_limits<double>::epsilon())
    {
        for (int iteration = 0;; iteration++) {
            arma::mat Fa = fock_matrix(Calpha);
            arma::mat Fb = fock_matrix(Cbeta);
            arma::vec epsilona;
            arma::mat Ca;
            arma::eig_sym(epsilona, Ca, Fa);
            Ca = Ca.head_cols(n_alpha);
            arma::vec epsilonb;
            arma::mat Cb;
            arma::eig_sym(epsilonb, Cb, Fb);
            Cb = Cb.head_cols(n_beta);
            arma::mat Pa_new = density_matrix(Ca);
            arma::mat Pb_new = density_matrix(Cb);
            arma::mat Pa = density_matrix(Calpha);
            arma::mat Pb = density_matrix(Cbeta);
            Calpha = Ca;
            Cbeta = Cb;
            if ((arma::norm(Pa_new - Pa, "inf") < tol) ||
                (arma::norm(Pb_new - Pb, "inf") < tol)) {
                break;
            }
        }
    }

    arma::mat core_hamiltonian() const
    {
        arma::mat Palpha = density_matrix(Calpha);
        arma::mat Pbeta = density_matrix(Cbeta);
        const arma::uword N = basis_count();
        arma::mat H = arma::zeros(N, N);
        arma::mat S = overlap_matrix();
        arma::uword i = 0;
        for (arma::uword a = 0; a < atoms.size(); a++) {
            Atom A = atoms[a];
            double gamma_AA = eri(A, A);
            for (CGAO Aorbital : A.orbitals) {
                H(i, i) = -0.5 * Aorbital.IA - (A.Z - 0.5) * gamma_AA;
                arma::uword j = 0;
                for (arma::uword b = 0; b < atoms.size(); b++) {
                    Atom B = atoms[b];
                    double gamma_AB = eri(A, B);
                    if (b != a) {
                        H(i, i) -= B.Z * gamma_AB;
                    }
                    for (CGAO Borbital : B.orbitals) {
                        if (i != j) {
                            H(i, j) =
                                0.5 * (Aorbital.beta + Borbital.beta) * S(i, j);
                        }
                        j++;
                    }
                }
                i++;
            }
        }
        return H;
    }

    arma::vec total_density_matrix() const
    {
        arma::mat P = density_matrix(Calpha) + density_matrix(Cbeta);
        arma::vec Ptot = arma::zeros(atoms.size());
        arma::uword i = 0;
        for (arma::uword a = 0; a < atoms.size(); a++) {
            Atom A = atoms[a];
            for (CGAO Aorbital : A.orbitals) {
                Ptot(a) += P(i, i);
                i++;
            }
        }
        return Ptot;
    }

    arma::mat fock_matrix(const arma::mat &C) const
    {
        arma::mat P = density_matrix(C);
        const arma::uword N = basis_count();
        arma::mat F = arma::zeros(N, N);
        arma::mat S = overlap_matrix();
        arma::vec Ptot = total_density_matrix();
        arma::uword i = 0;
        for (arma::uword a = 0; a < atoms.size(); a++) {
            Atom A = atoms[a];
            double gamma_AA = eri(A, A);
            for (CGAO Aorbital : A.orbitals) {
                F(i, i) = -0.5 * Aorbital.IA +
                          (Ptot(a) - A.Z - P(i, i) + 0.5) * gamma_AA;
                arma::uword j = 0;
                for (arma::uword b = 0; b < atoms.size(); b++) {
                    Atom B = atoms[b];
                    double gamma_AB = eri(A, B);
                    if (a != b) {
                        F(i, i) += (Ptot(b) - B.Z) * gamma_AB;
                    }
                    for (CGAO Borbital : B.orbitals) {
                        if (i != j) {
                            F(i, j) = 0.5 * (Aorbital.beta + Borbital.beta) *
                                          S(i, j) -
                                      P(i, j) * gamma_AB;
                        }
                        j++;
                    }
                }
                i++;
            }
        }
        return F;
    }

    arma::mat overlap_matrix() const
    {
        const arma::uword N = basis_count();
        arma::mat S = arma::zeros(0, 0);
        for (Atom A : atoms) {
            arma::mat S_i = arma::zeros(0, 0);
            for (Atom B : atoms) {
                arma::mat S_ij = overlap(A, B);
                S_i = arma::join_rows(S_i, S_ij);
            }
            S = arma::join_cols(S, S_i);
        }
        return S;
    }

    arma::cube overlap_matrix_grad() const
    {
        arma::cube S = arma::zeros(0, 0, 0);
        for (Atom A : atoms) {
            arma::cube S_i = arma::zeros(0, 0, 0);
            for (Atom B : atoms) {
                arma::cube S_ij = overlap_grad(A, B);
                S_i = arma::join_slices(S_i, S_ij);
            }
            if (S.n_cols == 0) {
                S = S_i;
            } else {
                for (arma::uword j = 0; j < S_i.n_cols; j++) {
                    S.insert_cols(S.n_cols, S_i.col(j));
                }
            }
        }
        return S;
    }

    double nuclear_repulsion_energy() const
    {
        double energy = 0.0;
        for (arma::uword a = 0; a < atoms.size(); a++) {
            Atom A = atoms[a];
            for (arma::uword b = 0; b < a; b++) {
                Atom B = atoms[b];
                energy += 27.211324570273 * A.Z * B.Z / arma::norm(A.r - B.r);
            }
        }
        return energy;
    }

    arma::vec nuclear_repulsion_energy_grad(arma::uword a) const
    {
        Atom A = atoms[a];
        arma::vec result = arma::zeros(3);
        for (arma::uword b = 0; b < atoms.size(); b++) {
            Atom B = atoms[b];
            if (b != a) {
                result -= 27.211324570273 * A.Z * B.Z * (A.r - B.r) /
                          pow(arma::norm(A.r - B.r), 3);
            }
        }
        return result;
    }

    double electronic_energy() const
    {
        arma::mat Pa = density_matrix(Calpha);
        arma::mat Pb = density_matrix(Cbeta);
        arma::mat H = core_hamiltonian();
        arma::mat Fa = fock_matrix(Calpha);
        arma::mat Fb = fock_matrix(Cbeta);
        return 0.5 * arma::accu(Pa % (H + Fa)) +
               0.5 * arma::accu(Pb % (H + Fb));
    }

    arma::vec electronic_energy_grad(arma::uword a) const
    {
        Atom A = atoms[a];
        arma::vec result = arma::zeros(3);
        arma::mat Pa = density_matrix(Calpha);
        arma::mat Pb = density_matrix(Cbeta);
        arma::mat P = Pa + Pb;
        arma::vec Ptot = total_density_matrix();
        arma::cube S_grad = overlap_matrix_grad();
        arma::uword j = 0;
        for (arma::uword b = 0; b < atoms.size(); b++) {
            Atom B = atoms[b];
            arma::vec gammaAB_RA = eri_grad(A, B);
            result += (Ptot(a) * Ptot(b) - B.Z * Ptot(a) - A.Z * Ptot(b)) *
                      gammaAB_RA;
            for (CGAO Borbital : B.orbitals) {
                arma::uword i = 0;
                for (arma::uword c = 0; c < atoms.size(); c++) {
                    Atom C = atoms[c];
                    for (CGAO Corbital : C.orbitals) {
                        if ((c == a) && (a != b)) {
                            result -= (pow(Pa(i, j), 2) + pow(Pb(i, j), 2)) *
                                      gammaAB_RA;
                            result -= (Corbital.beta + Borbital.beta) *
                                      P(i, j) *
                                      overlap_grad(Corbital, Borbital);
                        }
                        i++;
                    }
                }
                j++;
            }
        }
        return result;
    }

    double energy() const override
    {
        return electronic_energy() + nuclear_repulsion_energy();
    }
};

class PB
{
  public:
    arma::uvec n;
    double L;
    PB(const arma::uvec &N, double BOX_SIZE_ANGSTROM)
        : n(N), L(BOX_SIZE_ANGSTROM)
    {
    }

    arma::vec wavefunction(const arma::mat &grid_points) const
    {
        arma::mat shifted = (grid_points + L / 2.0) * (M_PI / L);
        arma::mat temp =
            arma::sin(shifted.each_col() % arma::conv_to<arma::vec>::from(n));
        return arma::prod(temp, 0).t() * pow(2.0 / L, 1.5);
    }

    double energy() const
    {
        return 27.211386246 * pow(M_PI * norm(n) / L, 2) / 2.0;
    }

    bool operator<(const PB &other) const { return energy() < other.energy(); }
};

class DFT : public Simulation
{
  public:
    int n_alpha;
    int n_beta;
    double L;
    double E_cutoff;
    int N_grid;
    arma::mat grid_points;
    arma::mat Calpha;
    arma::mat Cbeta;
    double N;
    std::vector<PB> basis;
    std::vector<PB> auxilliary_basis;
    arma::mat grid_wavefunction;
    arma::mat grid_auxilliary_wavefunction;
    double grid_weight;
    int include;
    bool density_fitting;
    double tol;
    DFT(const std::vector<Atom> &atoms, const int N_ALPHA, const int N_BETA,
        const double BOX_SIZE_ANGSTROM, const double KINETIC_ENERGY_CUTOFF_EV,
        const int NUMBER_GRID_POINTS, const int INCLUDE = 3,
        const bool DENSITY_FITTING = true,
        const double TOL = std::numeric_limits<double>::epsilon())
        : Simulation(atoms), n_alpha(N_ALPHA), n_beta(N_BETA),
          L(BOX_SIZE_ANGSTROM / 0.529177210544),
          E_cutoff(KINETIC_ENERGY_CUTOFF_EV / 27.211386246),
          N_grid(NUMBER_GRID_POINTS), include(INCLUDE),
          density_fitting(DENSITY_FITTING), tol(TOL)
    {
        grid_weight = pow(L / N_grid, 3);
        double shift = L / 2.0 * (1.0 - 1.0 / N_grid);
        arma::rowvec grid_points_1d =
            arma::linspace<arma::rowvec>(-shift, shift, N_grid);
        arma::uword N_grid2 = N_grid * N_grid;
        arma::mat X = arma::repelem(grid_points_1d, 1, N_grid2);
        arma::mat Y = arma::repelem(grid_points_1d, 1, N_grid);
        Y = arma::repmat(Y, 1, N_grid);
        arma::mat Z = arma::repmat(grid_points_1d, 1, N_grid2);
        grid_points = arma::join_cols(X, Y, Z);

        double N = sqrt(E_cutoff * 2 * pow(L / M_PI, 2));

        basis = std::vector<PB>();
        auxilliary_basis = std::vector<PB>();
        arma::uword inc = 1;
        bool odd_only =
            atoms.size() == 1 && atoms[0].E <= 2 &&
            arma::norm(atoms[0].r) <= std::numeric_limits<double>::epsilon();
        int N_aux = 2 * N;
        for (arma::uword i = 1; i <= N_aux; i++) {
            for (arma::uword j = 1; j <= N_aux; j++) {
                for (arma::uword k = 1; k <= N_aux; k++) {
                    if ((i * i + j * j + k * k) <= N_aux * N_aux) {
                        auxilliary_basis.push_back(
                            PB(arma::uvec({i, j, k}), L));
                        if ((i * i + j * j + k * k) <= N * N &&
                            (!odd_only ||
                             (i % 2 == 1 && j % 2 == 1 && k % 2 == 1))) {
                            basis.push_back(PB(arma::uvec({i, j, k}), L));
                        }
                    }
                }
            }
        }

        std::sort(basis.begin(), basis.end());
        std::sort(auxilliary_basis.begin(), auxilliary_basis.end());

        Calpha = arma::zeros(basis.size(), n_alpha);
        Cbeta = arma::zeros(basis.size(), n_beta);

        grid_wavefunction = arma::zeros(grid_points.n_cols, basis.size());
        for (arma::uword mu = 0; mu < basis.size(); mu++) {
            grid_wavefunction.col(mu) = basis[mu].wavefunction(grid_points);
        }
        grid_auxilliary_wavefunction =
            arma::zeros(grid_points.n_cols, auxilliary_basis.size());
        for (arma::uword mu = 0; mu < auxilliary_basis.size(); mu++) {
            grid_auxilliary_wavefunction.col(mu) =
                auxilliary_basis[mu].wavefunction(grid_points);
        }
        std::cout << "Box lattice vectors (Bohr):   " << arma::eye(3, 3) * L
                  << std::endl;
        std::cout << "Cutoff energy (eV): " << E_cutoff * 27.211386246
                  << std::endl;
        std::cout << "Number of basis functions: " << basis.size() << std::endl;
        std::cout << "Number of grid points: " << N_grid << " x " << N_grid
                  << " x " << N_grid << std::endl;
        std::cout << "Number of electrons: " << n_alpha + n_beta << std::endl;
        std::cout << "Real space density initialized (zeros)" << std::endl;
        std::cout << "Will finish constructing the kinetic energy part of the "
                     "Hamiltonian later"
                  << std::endl;
        std::cout << "Finished evaluating the basis set at each grid point"
                  << std::endl;
    }

    void converge(const bool debug = false)
    {
        for (int iteration = 0;; iteration++) {
            arma::mat Fa = fock_matrix(Calpha);
            arma::mat Fb = fock_matrix(Cbeta);
            arma::vec epsilona;
            arma::mat Ca;
            arma::eig_sym(epsilona, Ca, Fa);
            Ca = Ca.head_cols(n_alpha);
            arma::vec epsilonb;
            arma::mat Cb;
            arma::eig_sym(epsilonb, Cb, Fb);
            Cb = Cb.head_cols(n_beta);
            arma::mat Pa_new = density_matrix(Ca);
            arma::mat Pb_new = density_matrix(Cb);
            arma::mat Pa = density_matrix(Calpha);
            arma::mat Pb = density_matrix(Cbeta);
            Calpha = Ca;
            Cbeta = Cb;

            if (debug) {
                std::cout << "Iteration:" << iteration << std::endl;
                std::cout << "The total number of electron is "
                          << n_alpha + n_beta << std::endl;
                std::cout << "Energy of occupied orbital (alpha) 0: "
                          << epsilona(0) << std::endl;
                std::cout << "Kinetic energy: " << kinetic_energy()
                          << std::endl;
                std::cout << "Hartree energy: " << hartree_energy()
                          << std::endl;
                std::cout << "External energy: " << external_energy()
                          << std::endl;
                std::cout << "Exchange-correlation energy (alpha): "
                          << exchange_correlation_energy(Calpha) << std::endl;
                std::cout << "Energy of occupied orbital (beta) 0: "
                          << epsilonb(0) << std::endl;
                std::cout << "Exchange-correlation energy (beta): "
                          << exchange_correlation_energy(Cbeta) << std::endl;
                std::cout << "Exchange-correlation energy (total): "
                          << exchange_correlation_energy(Calpha) +
                                 exchange_correlation_energy(Cbeta)
                          << std::endl;
                std::cout << "Total energy: " << energy() << std::endl;
                for (arma::uword mu = 0; mu < basis.size(); mu++) {
                    double contribution = pow(Calpha(mu, 0), 2);
                    if (contribution >= 0.01) {
                        std::cout << "Basis contribution:" << basis[mu].n.t();
                        std::cout << "to orbital (alpha) 0 is " << contribution
                                  << std::endl;
                    }
                }
            }
            double err_a = arma::norm(Pa_new - Pa, "inf");
            double err_b = arma::norm(Pb_new - Pb, "inf");
            if ((err_a < tol) && (err_b < tol)) {
                break;
            }
        }
    }

    arma::vec density(const arma::mat &C) const
    {
        arma::mat phi = grid_wavefunction * C;
        return arma::sum(phi % phi, 1);
    }

    arma::mat kinetic_matrix() const
    {
        arma::vec T = arma::zeros(basis.size());
        for (arma::uword mu = 0; mu < basis.size(); mu++) {
            T(mu) = pow(M_PI * arma::norm(basis[mu].n) / L, 2.0) / 2.0;
        }
        return arma::diagmat(T);
    }

    arma::mat grid_integration(arma::vec &V_r) const
    {
        arma::mat V = arma::zeros(basis.size(), basis.size());
        for (arma::uword mu = 0; mu < basis.size(); mu++) {
            for (arma::uword nu = 0; nu < basis.size(); nu++) {
                V(mu, nu) =
                    grid_weight * arma::accu(grid_wavefunction.col(mu) % V_r %
                                             grid_wavefunction.col(nu));
            }
        }
        return V;
    }

    arma::mat external_potential_matrix() const
    {
        arma::vec Vext = arma::zeros(grid_points.n_cols);
        if (density_fitting) {
            arma::vec rho = density(Calpha) + density(Cbeta);
            arma::vec CH = arma::zeros(auxilliary_basis.size());
            for (arma::uword mu = 0; mu < auxilliary_basis.size(); mu++) {
                arma::mat atom_points = arma::zeros(3, atoms.size());
                for (arma::uword a = 0; a < atoms.size(); a++) {
                    Atom A = atoms[a];
                    atom_points.col(a) = A.r;
                }
                arma::vec omegaRA =
                    auxilliary_basis[mu].wavefunction(atom_points);
                double sum = 0.0;
                for (arma::uword a = 0; a < atoms.size(); a++) {
                    Atom A = atoms[a];
                    sum += A.E * omegaRA(a);
                }
                CH(mu) = -4.0 / M_PI *
                         pow(L / arma::norm(auxilliary_basis[mu].n), 2.0) * sum;
                Vext += grid_auxilliary_wavefunction.col(mu) * CH(mu);
            }
        } else {
            for (arma::uword i = 0; i < grid_points.n_cols; i++) {
                for (arma::uword a = 0; a < atoms.size(); a++) {
                    Atom A = atoms[a];
                    Vext(i) -= A.E / arma::norm(grid_points.col(i) - A.r);
                }
            }
        }
        return grid_integration(Vext);
    }

    arma::mat hartree_potential_matrix() const
    {
        arma::vec VH = arma::zeros(grid_points.n_cols);
        arma::vec rho = density(Calpha) + density(Cbeta);
        if (density_fitting) {
            arma::vec CH = arma::zeros(auxilliary_basis.size());
            for (arma::uword mu = 0; mu < auxilliary_basis.size(); mu++) {
                arma::vec omega = grid_auxilliary_wavefunction.col(mu);
                CH(mu) =
                    4.0 * M_PI *
                    pow(L / M_PI / arma::norm(auxilliary_basis[mu].n), 2.0) *
                    grid_weight * arma::accu(omega % rho);
                VH += omega * CH(mu);
            }
        } else {
            arma::cube V(VH.begin(), N_grid, N_grid, N_grid);
            arma::cube rho_mat(rho.begin(), N_grid, N_grid, N_grid);
            for (int iteration = 0;; iteration++) {
                arma::cube V_old = V;
                V = 4.0 * M_PI * pow(L / N_grid, 2.0) * rho_mat / 6.0;
                V.subcube(0, 0, 0, N_grid - 2, N_grid - 1, N_grid - 1) +=
                    V_old.subcube(1, 0, 0, N_grid - 1, N_grid - 1, N_grid - 1) /
                    6.0;
                V.subcube(1, 0, 0, N_grid - 1, N_grid - 1, N_grid - 1) +=
                    V_old.subcube(0, 0, 0, N_grid - 2, N_grid - 1, N_grid - 1) /
                    6.0;
                V.subcube(0, 0, 0, N_grid - 1, N_grid - 2, N_grid - 1) +=
                    V_old.subcube(0, 1, 0, N_grid - 1, N_grid - 1, N_grid - 1) /
                    6.0;
                V.subcube(0, 1, 0, N_grid - 1, N_grid - 1, N_grid - 1) +=
                    V_old.subcube(0, 0, 0, N_grid - 1, N_grid - 2, N_grid - 1) /
                    6.0;
                V.subcube(0, 0, 0, N_grid - 1, N_grid - 1, N_grid - 2) +=
                    V_old.subcube(0, 0, 1, N_grid - 1, N_grid - 1, N_grid - 1) /
                    6.0;
                V.subcube(0, 0, 1, N_grid - 1, N_grid - 1, N_grid - 1) +=
                    V_old.subcube(0, 0, 0, N_grid - 1, N_grid - 1, N_grid - 2) /
                    6.0;
                if (arma::abs(V - V_old).max() < tol) {
                    std::cout << "Poisson solver converged in " << iteration
                              << " iterations" << std::endl;
                    VH = arma::vectorise(V);
                    break;
                }
            }
        }
        return grid_integration(VH);
    }

    arma::mat exchange_correlation_matrix(const arma::mat &C) const
    {
        arma::vec Vxc = -arma::pow((3.0 / M_PI) * density(C), 1.0 / 3);
        return grid_integration(Vxc);
    }

    arma::mat fock_matrix(const arma::mat &C) const
    {
        arma::mat F = kinetic_matrix();
        if (include >= 1) {
            arma::mat Vext = external_potential_matrix();
            F += Vext;
        }
        if (include >= 2) {
            F += hartree_potential_matrix();
        }
        if (include >= 3) {
            F += exchange_correlation_matrix(C);
        }
        return F;
    }

    arma::mat overlap_matrix() const
    {
        return arma::eye(basis.size(), basis.size());
    }

    double kinetic_energy() const
    {
        arma::mat P = density_matrix(Calpha) + density_matrix(Cbeta);
        arma::mat T = kinetic_matrix();
        return arma::trace(P * T);
    }

    double hartree_energy() const
    {
        arma::mat P = density_matrix(Calpha) + density_matrix(Cbeta);
        arma::mat VH = hartree_potential_matrix();
        return 0.5 * arma::trace(P * VH);
    }

    double external_energy() const
    {
        arma::mat P = density_matrix(Calpha) + density_matrix(Cbeta);
        arma::mat Vext = external_potential_matrix();
        return arma::trace(P * Vext);
    }

    double exchange_correlation_energy(const arma::mat &C) const
    {
        arma::mat f = arma::pow(density(C), 4.0 / 3);
        return -0.75 * pow(3.0 / M_PI, 1.0 / 3) * grid_weight * arma::accu(f);
    }

    double energy() const override
    {
        double E = kinetic_energy();
        if (include >= 1) {
            E += external_energy();
        }
        if (include >= 2) {
            E += hartree_energy();
        }
        if (include >= 3) {
            E += exchange_correlation_energy(Calpha) +
                 exchange_correlation_energy(Cbeta);
        }
        return E;
    }
};