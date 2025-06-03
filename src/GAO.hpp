#pragma once

#include <armadillo>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "tools.hpp"

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
    return eri(A.orbitals[0], B.orbitals[0]);
}

arma::vec eri_grad(const Atom &A, const Atom &B)
{
    return eri_grad(A.orbitals[0], B.orbitals[0]);
}