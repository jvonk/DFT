#pragma once 

#include <armadillo> 
#include <cmath> 
#include <iostream> 
#include <iomanip>
#include <math.h>

double calc_error(arma::mat exact, arma::mat value) {
    double total = 0;
    for (int i = 0; i < exact.n_cols; i++) {
        for (int j = 0; j < 3; j++) {
            total += pow(value(j, i) - exact(j, i), 2);
        }
    }
    return sqrt(total);
}
double norm2(arma::mat value) {
    double total = 0;
    for (int i = 0; i < value.n_cols; i++) {
        for (int j = 0; j < value.n_rows; j++) {
            total += pow(value(j, i), 2);
        }
    }
    return sqrt(total);
}


int factorial(int n) {
    int product = 1;
    for (int i = 1; i <= n; i++) {
        product *= i;
    }
    return product;
}
int double_factorial(int n) {
    int product = 1;
    for (int i = n; i > 0; i -= 2) {
        product *= i;
    }
    return product;
}
double binomial(int m, int n) {
    return factorial(m) / (factorial(n) * factorial(m - n));
}

class Quadrature {
public:
    int refinement;
    virtual double refine() = 0;
    double operator()(const double tol, const int MAXSTEPS = 40) {
        double oldvalue;
        for (int i = 0; i < MAXSTEPS; i++) {
            double value = refine();
            if ((i > 0) && ((fabs(value - oldvalue) < tol) || (value == 0 && oldvalue == 0))) {
                return value;
            }
            oldvalue = value;
        }
        throw("Quadrature did not converge, too many steps.");
    }
};

template<class T>
class Trapezoid : Quadrature {
private:
    double lower, upper, value;
public:
    T& f;
    Trapezoid() {}
    Trapezoid(T& F, const double LOWER, const double UPPER) :
        f(F), lower(LOWER), upper(UPPER) {
        refinement = 0;
    }
    double refine() {
        if (refinement++ == 0) {
            value = (upper - lower) * (f(lower) + f(upper)) / 2.0;
        }
        else {
            int nsteps = 1 << (refinement - 2);
            double del = (upper - lower) / nsteps;
            double x = lower + del / 2.0;
            double sum = 0.0;
            for (int i = 0; i < nsteps; i++, x+=del) {
                sum += f(x);
            }
            value = (value + (upper - lower) * sum / nsteps) / 2.0;
        }
        return value;
    }
    double operator()(const double tol, const int MAXSTEPS = 20) {
        return Quadrature::operator()(tol, MAXSTEPS);
    }
};

class Function {
public:
    virtual double operator()(double x) = 0;
};

class DoubleGaussian : Function {
private:
    double XA, XB, alpha, beta;
    int lA, lB;
public:
    DoubleGaussian(const double XA_in, const int lA, const double alpha, const double XB, const int lB, const double beta) :
        XA(XA), lA(lA), alpha(alpha), XB(XB), lB(lB), beta(beta) {}
    double operator()(double x) {
        double dxA = x - XA;
        double dxB = x - XB;
        return pow(dxA, lA) * pow(dxB, lB) * exp(-alpha * dxA * dxA - beta * dxB * dxB);
    }
};

int n_functions(const int l) {
    return (l + 1) * (l + 2) / 2;
}

arma::umat lmat(const int l) {
    int count = n_functions(l);
    arma::umat result(3, count);
    for (int i = 0; i <= l; i++) {
        for (int j = 0; (j <= l) && (i + j <= l); j++) {
            count--;
            result(0, count) = i;
            result(1, count) = j;
            result(2, count) = l - i - j;
        }
    }
    return result;
}
class GAO {
public:
    arma::vec X;
    double alpha;
    arma::uvec l;
    GAO() {
        X = arma::vec(3);
        l = arma::uvec(3);
    }
    GAO(const arma::vec& X, const double alpha, const arma::uvec l) :
        X(X), alpha(alpha), l(l) {}
    friend std::istream & operator >> (std::istream &in, GAO& basis);
    friend std::ostream & operator << (std::ostream &out, const GAO& basis);
};

// std::istream & operator >> (std::istream &in, GAO& basis) {
//     in >> basis.X(0) >> basis.X(1) >> basis.X(2) >> basis.alpha >> basis.l;
//     return in;
// }
// std::ostream & operator << (std::ostream &out, const GAO& basis) {
//     out << std::setprecision(2) << std::fixed;
//     out << "This shell info: R( " << basis.X(0) << ", " << basis.X(1) << ", " << basis.X(2) << "), with angular momentum: " << shell.l << ", coefficient: " << shell.alpha;
//     out << std::defaultfloat;
//     return out;
// }

double overlap(const GAO& A, const GAO& B, const int dir) {
    double alpha = A.alpha;
    double beta = B.alpha;
    arma::vec RP = (alpha * A.X + beta * B.X) / (alpha + beta);
    int lA = A.l(dir);
    int lB = B.l(dir);
    double sum = 0.0;
    double XA = A.X(dir);
    double XB = B.X(dir);
    double XP = RP(dir);
    double prefactor = exp( - alpha * beta * (XA - XB) * (XA - XB) / (alpha + beta)) * sqrt(M_PI / (alpha + beta));
    for (int i = 0; i <= lA; i++) {
        for (int j = 0; j <= lB; j++) {
            if ((i + j) % 2 == 0) {
                sum += binomial(lA, i) * binomial(lB, j) * double_factorial(i + j - 1) * pow(XP - XA, lA - i) * pow(XP - XB, lB - j) / pow(2.0 * (alpha + beta), (i + j) / 2.0);
            }
        }
    }
    return prefactor * sum;
}

double overlap(const GAO& A, const GAO& B) {
    double result = 1.0;
    for (int dir = 0; dir < 3; dir++) {
        result *= overlap(A, B, dir);
    }
    return result;
}

arma::vec overlap_grad(const GAO& A, const GAO& B) {
    double alpha = A.alpha;
    double beta = B.alpha;
    arma::vec RP = (alpha * A.X + beta * B.X) / (alpha + beta);
    arma::vec result = arma::ones(3);
    if (norm2(A.X - B.X) < 1e-10) {
        return arma::zeros(3);
    }
    for (int dir = 0; dir < 3; dir++) {
        int lA = A.l(dir);
        int lB = B.l(dir);
        double XA = A.X(dir);
        double XB = B.X(dir);
        double XP = RP(dir);
        double prefactor = exp( - alpha * beta * (XA - XB) * (XA - XB) / (alpha + beta)) * sqrt(M_PI / (alpha + beta));
        double sum = 0.0;
        for (int i = 0; i <= lA + 1; i++) {
            for (int j = 0; j <= lB; j++) {
                if ((i + j) % 2 == 0) {
                    if (i <= lA - 1) {
                        sum -= lA * binomial(lA - 1, i) * binomial(lB, j) * double_factorial(i + j - 1) * pow(XP - XA, lA - 1 - i) * pow(XP - XB, lB - j) / pow(2.0 * (alpha + beta), (i + j) / 2.0);
                    }
                    sum += 2 * alpha * binomial(lA + 1, i) * binomial(lB, j) * double_factorial(i + j - 1) * pow(XP - XA, lA + 1 - i) * pow(XP - XB, lB - j) / pow(2.0 * (alpha + beta), (i + j) / 2.0);
                }
            }
        }
        result(dir) *= prefactor * sum;
        for (int other_dir = 0; other_dir < 3; other_dir++) {
            if (other_dir != dir) {
                result(dir) *= overlap(A, B, other_dir);
            }
        }
    }
    return -result;
}

double eri(const GAO& A, const GAO& Ap, const GAO& B, const GAO& Bp) {
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

arma::vec eri_grad(const GAO& A, const GAO& Ap, const GAO& B, const GAO& Bp) {
    double sigmaA = 1.0 / (A.alpha + Ap.alpha);
    double sigmaB = 1.0 / (B.alpha + Bp.alpha);
    double V2 = 1.0 / (sigmaA + sigmaB);
    double U = pow(M_PI * sigmaA * M_PI * sigmaB, 1.5);
    double T = V2 * arma::dot(A.X - B.X, A.X - B.X);
    if (T == 0.0) {
        return arma::zeros(3);
    } else {
        return 27.211324570273 * U * (A.X - B.X) / pow(arma::norm(A.X - B.X), 2) * (- std::erf(sqrt(T)) / arma::norm(A.X - B.X) + 2.0 * sqrt(V2 / M_PI) * exp(-T));
    }
    
}

class CGAO {
public:
    arma::vec X;
    arma::uvec l;
    arma::vec d;
    arma::vec N;
    double h;
    double IA;
    int beta;
    std::vector<GAO> basis;
    CGAO() {
        X = arma::vec(3);
        l = arma::uvec(3);
        d = arma::vec(3);
        N = arma::vec(3);
        basis = std::vector<GAO>();
    }
    CGAO(const arma::vec X, const arma::uvec l, const int atomicNumber) :
        X(X), l(l) {
        arma::vec alpha;
        if (atomicNumber == 1 && arma::sum(l) == 0) {
            alpha = {3.42525091, 0.62391373, 0.16885540};
            d = {0.15432897, 0.53532814, 0.44463454};
            h = -13.6;
            IA = 2.0*7.176;
            beta = -9;
        } else if (atomicNumber == 6 && arma::sum(l) == 0) {
            alpha = {2.94124940, 0.68348310, 0.22228990};
            d = {-0.09996723, 0.39951283, 0.70011547};
            h = -21.4;
            IA = 2.0*14.051;
            beta = -21;
        } else if (atomicNumber == 6 && arma::sum(l) == 1) {
            alpha = {2.94124940, 0.68348310, 0.22228990};
            d = {0.15591627, 0.60768372, 0.39195739};
            h = -11.4;
            IA = 2.0*5.572;
            beta = -21;
        } else if (atomicNumber == 7 && arma::sum(l) == 0) {
            alpha = {3.78045590, 0.87849660, 0.28571440};
            d = {-0.09996723, 0.39951283, 0.70011547};
            h = -21.4;
            IA = 2.0*19.316;
            beta = -25;
        } else if (atomicNumber == 7 && arma::sum(l) == 1) {
            alpha = {3.78045590, 0.87849660, 0.28571440};
            d = {0.15591627, 0.60768372, 0.39195739};
            h = -11.4;
            IA = 2.0*7.275;
            beta = -25;
        } else if (atomicNumber == 8 && arma::sum(l) == 0) {
            alpha = {5.03315130, 1.16959610, 0.38038900};
            d = {-0.09996723, 0.39951283, 0.70011547};
            h = -21.4;
            IA = 2.0*25.390;
            beta = -31;
        } else if (atomicNumber == 8 && arma::sum(l) == 1) {
            alpha = {5.03315130, 1.16959610, 0.38038900};
            d = {0.15591627, 0.60768372, 0.39195739};
            h = -11.4;
            IA = 2.0*9.111;
            beta = -31;
        } else if (atomicNumber == 9 && arma::sum(l) == 0) {
            alpha = {6.46480320, 1.50228120, 0.48858850};
            d = {-0.09996723, 0.39951283, 0.70011547};
            h = -21.4;
            IA = 2.0*32.272;
            beta = -39;
        } else if (atomicNumber == 9 && arma::sum(l) == 1) {
            alpha = {6.46480320, 1.50228120, 0.48858850};
            d = {0.15591627, 0.60768372, 0.39195739};
            h = -11.4;
            IA = 2.0*11.080;
            beta = -39;
        } else {
            throw std::invalid_argument("Element or orbital not supported!");
        }
        N = arma::vec(alpha.n_elem);
        for (int i = 0; i < alpha.n_elem; i++) {
            GAO element = GAO(X, alpha(i), l);
            basis.push_back(element);
            N(i) = pow(overlap(element, element), -0.5);
        }
    }
};

double overlap(const CGAO& A, const CGAO& B) {
    double result = 0.0;
    for (int i = 0; i < A.basis.size(); i++) {
        for (int j = 0; j < B.basis.size(); j++) {
            result += A.d(i) * A.N(i) * B.d(j) * B.N(j) * overlap(A.basis[i], B.basis[j]);
        }
    }
    return result;
}

arma::vec overlap_grad(const CGAO& A, const CGAO& B) {
    arma::vec result = arma::zeros(3);
    for (int i = 0; i < A.basis.size(); i++) {
        for (int j = 0; j < B.basis.size(); j++) {
            result += A.d(i) * A.N(i) * B.d(j) * B.N(j) * overlap_grad(A.basis[i], B.basis[j]);
        }
    }
    return result;
}

double eri(const CGAO& A, const CGAO& B) {
    double result = 0.0;
    arma::vec Adp = A.d % A.N;
    arma::vec Bdp = B.d % B.N;
    for (int k = 0; k < 3; k++) {
        for (int kp = 0; kp < 3; kp++) {
            for (int l = 0; l < 3; l++) {
                for (int lp = 0; lp < 3; lp++) {
                    result += Adp(k) * Adp(kp) * Bdp(l) * Bdp(lp) * eri(A.basis[k], A.basis[kp], B.basis[l], B.basis[lp]);
                }
            }
        }
    }
    return result;
}

arma::vec eri_grad(const CGAO& A, const CGAO& B) {
    arma::vec result = arma::zeros(3);
    arma::vec Adp = A.d % A.N;
    arma::vec Bdp = B.d % B.N;
    for (int k = 0; k < 3; k++) {
        for (int kp = 0; kp < 3; kp++) {
            for (int l = 0; l < 3; l++) {
                for (int lp = 0; lp < 3; lp++) {
                    result += Adp(k) * Adp(kp) * Bdp(l) * Bdp(lp) * eri_grad(A.basis[k], A.basis[kp], B.basis[l], B.basis[lp]);
                }
            }
        }
    }
    return result;
}

class Atom {
public:
    int E;
    int Z;
    arma::vec r;
    std::vector<CGAO> orbitals;
    Atom(int E, const arma::vec& r) : E(E), r(r) {
        orbitals = std::vector<CGAO>();
        if (E == 1 || E == 6 || E == 7 || E == 8 || E == 9) {
            arma::umat ls = lmat(0);
            for (int i = 0; i < ls.n_cols; i++) {
                orbitals.push_back(CGAO(r, ls.col(i), E));
            }
            if (E != 1) {
                ls = lmat(1);
                for (int i = 0; i < ls.n_cols; i++) {
                    orbitals.push_back(CGAO(r, ls.col(i), E));
                }
                Z = E - 2;
            } else {
                Z = 1;
            }
        } else {
            throw std::invalid_argument("Element not supported!");
        }
    }
    void move(const arma::vec& forces, const double step) {
        r += step * forces;
        orbitals = std::vector<CGAO>();
        if (E == 1 || E == 6 || E == 7 || E == 8 || E == 9) {
            arma::umat ls = lmat(0);
            for (int i = 0; i < ls.n_cols; i++) {
                orbitals.push_back(CGAO(r, ls.col(i), E));
            }
            if (E != 1) {
                ls = lmat(1);
                for (int i = 0; i < ls.n_cols; i++) {
                    orbitals.push_back(CGAO(r, ls.col(i), E));
                }
                Z = E - 2;
            } else {
                Z = 1;
            }
        } else {
            throw std::invalid_argument("Element not supported!");
        }
    }
};

std::ostream& operator << (std::ostream& os, const Atom& atom) {
    return (os << atom.E << "(" << atom.r[0] << ", " << atom.r[1] << ", " << atom.r[2] << ")");
}

arma::mat overlap(const Atom& A, const Atom& B) {
    arma::mat result = arma::zeros(A.orbitals.size(), B.orbitals.size());
    for (int i = 0; i < A.orbitals.size(); i++) {
        for (int j = 0; j < B.orbitals.size(); j++) {
            result(i, j) = overlap(A.orbitals[i], B.orbitals[j]);
        }
    }
    return result;
}

arma::cube overlap_grad(const Atom& A, const Atom& B) {
    arma::cube result = arma::zeros(3, A.orbitals.size(), B.orbitals.size());
    for (int i = 0; i < A.orbitals.size(); i++) {
        for (int j = 0; j < B.orbitals.size(); j++) {
            result.slice(j).col(i) = overlap_grad(A.orbitals[i], B.orbitals[j]);
        }
    }
    return result;
}

double eri(const Atom& A, const Atom& B) {
    // Only sum over valence s orbital centered on atoms A and B
    return eri(A.orbitals[0], B.orbitals[0]);
}

arma::vec eri_grad(const Atom& A, const Atom& B) {
    return eri_grad(A.orbitals[0], B.orbitals[0]);
}

arma::mat density(arma::mat C, int p) {
    arma::mat P = arma::zeros(C.n_rows, C.n_rows);
    for (int i = 0; i < p; i++) {
        P += C.col(i) * C.col(i).t();
    }
    return P;
}

class Simulation {
public:
    std::vector<Atom> atoms;
    std::string kind;
    int n_alpha;
    int n_beta;
    double sigma;
    double epsilon;
    arma::mat forces;
    arma::mat Calpha;
    arma::mat Cbeta;

    Simulation(const std::vector<Atom> atoms, const std::string KIND = "CNDO2", const int N_ALPHA = 0, const int N_BETA = 0, const double SIGMA = 2.951, const double EPSILON = 5.29) : 
        atoms(atoms), kind(KIND), n_alpha(N_ALPHA), n_beta(N_BETA), sigma(SIGMA), epsilon(EPSILON) {
        int N = basis_count();
        Calpha = arma::zeros(N, N);
        Cbeta = arma::zeros(N, N);
    }

    Simulation(const Simulation& sim) {
        atoms = sim.atoms;
        kind = sim.kind;
        n_alpha = sim.n_alpha;
        n_beta = sim.n_beta;
        sigma = sim.sigma;
        epsilon = sim.epsilon;
        forces = sim.forces;
        Calpha = sim.Calpha;
        Cbeta = sim.Cbeta;
        // int N = basis_count();
        // Calpha = arma::zeros(N, N);
        // Cbeta = arma::zeros(N, N);
    }

    Simulation(const Simulation& sim, arma::mat forces, double step, const double tol = 1e-10) {
        atoms = sim.atoms;
        kind = sim.kind;
        n_alpha = sim.n_alpha;
        n_beta = sim.n_beta;
        sigma = sim.sigma;
        epsilon = sim.epsilon;
        forces = sim.forces;
        Calpha = sim.Calpha;
        Cbeta = sim.Cbeta;
        // int N = basis_count();
        // Calpha = arma::zeros(N, N);
        // Cbeta = arma::zeros(N, N);
        double total = 0.0;
        for (int i = 0; i < atoms.size(); i++) {
            for (int j = 0; j < 3; j++) {
                total += pow(forces(j, i), 2);
            }
        }
        step /= sqrt(total);
        for (int i = 0; i < atoms.size(); i++) {
            atoms[i].move(forces.col(i), step);
        }
        if (kind == "CNDO2") {
            int num_basis_functions = basis_count();
            for (int iteration = 0; ; iteration++) {
                arma::mat Fa = fock_matrix(true);
                arma::mat Fb = fock_matrix(false);
                arma::vec epsilona;
                arma::mat Ca;
                arma::eig_sym(epsilona, Ca, Fa);
                arma::vec epsilonb;
                arma::mat Cb;
                arma::eig_sym(epsilonb, Cb, Fb);
                arma::mat Pa_new = density(Ca, n_alpha);
                arma::mat Pb_new = density(Cb, n_beta);
                arma::mat Pa = density(Calpha, n_alpha);
                arma::mat Pb = density(Cbeta, n_beta);
                Calpha = Ca;
                Cbeta = Cb;
                if ((arma::norm(Pa_new - Pa, "inf") < tol) || (arma::norm(Pb_new - Pb, "inf") < tol)) {
                    break;
                }
                int i = 0;
                arma::vec Ptot = arma::zeros(atoms.size());
                for (int a = 0; a < atoms.size(); a++) {
                    Atom A = atoms[a];
                    for (CGAO Aorbital : A.orbitals) {
                        Ptot(a) += Pa_new(i, i);
                        Ptot(a) += Pb_new(i, i);
                        i++;
                    }
                }
            }
        }
    }

    int basis_count() {
        int a = 0;
        int b = 0;
        for (Atom atom : atoms) {
            if (atom.E == 1) {
                b++;
            } else if (atom.E == 6 || atom.E == 7 || atom.E == 8 || atom.E == 9) {
                a++;
            }
        }
        return 4 * a + b;
    }

    int electron_count() {
        const int N = basis_count();
        if (N % 2 != 0) {
            throw std::invalid_argument("Odd number of valence electrons!");
        }
        return N / 2;
    }

    arma::mat core_hamiltonian() {
        arma::mat Palpha = density(Calpha, n_alpha);
        arma::mat Pbeta = density(Cbeta, n_beta);
        const int N = basis_count();
        arma::mat H = arma::zeros(N, N);
        arma::mat S = overlap_matrix();
        int i = 0;
        for (int a = 0; a < atoms.size(); a++) {
            Atom A = atoms[a];
            double gamma_AA = eri(A, A);
            for (CGAO Aorbital : A.orbitals) {
                H(i, i) = -0.5 * Aorbital.IA - (A.Z - 0.5) * gamma_AA;
                int j = 0;
                for (int b = 0; b < atoms.size(); b++) {
                    Atom B = atoms[b];
                    double gamma_AB = eri(A, B);
                    if (b != a) {
                        H(i, i) -= B.Z * gamma_AB;
                    }
                    for (CGAO Borbital : B.orbitals) {
                        if (i != j) {
                            H(i, j) = 0.5 * (Aorbital.beta + Borbital.beta) * S(i, j);
                        }
                        j++;
                    }
                }
                i++;
            }
        }
        return H;
    }
    
    arma::vec total_density_matrix() {
        arma::mat P = density(Calpha, n_alpha) + density(Cbeta, n_beta);
        arma::vec Ptot = arma::zeros(atoms.size());
        int i = 0;
        for (int a = 0; a < atoms.size(); a++) {
            Atom A = atoms[a];
            for (CGAO Aorbital : A.orbitals) {
                Ptot(a) += P(i, i);
                i++;
            }
        }
        return Ptot;
    }

    arma::mat fock_matrix(const bool alpha = true) {
        arma::mat Palpha, Pbeta;
        if (alpha) {
            Palpha = density(Calpha, n_alpha);
            Pbeta = density(Cbeta, n_beta);
        } else {
            Palpha = density(Cbeta, n_beta);
            Pbeta = density(Calpha, n_alpha);
        }
        const int N = basis_count();
        arma::mat F = arma::zeros(N, N);
        arma::mat S = overlap_matrix();
        arma::vec Ptot = total_density_matrix();
        int i = 0;
        for (int a = 0; a < atoms.size(); a++) {
            Atom A = atoms[a];
            double gamma_AA = eri(A, A);
            for (CGAO Aorbital : A.orbitals) {
                F(i, i) = -0.5 * Aorbital.IA + (Ptot(a) - A.Z - Palpha(i, i) + 0.5) * gamma_AA;
                int j = 0;
                for (int b = 0; b < atoms.size(); b++) {
                    Atom B = atoms[b];
                    double gamma_AB = eri(A, B);
                    if (a != b) {
                        F(i, i) += (Ptot(b) - B.Z) * gamma_AB;
                    }
                    for (CGAO Borbital : B.orbitals) {
                        if (i != j) {
                            F(i, j) = 0.5 * (Aorbital.beta + Borbital.beta) * S(i, j) - Palpha(i, j) * gamma_AB;
                        }
                        j++;
                    }
                }
                i++;
            }
        }
        return F;
    }

    arma::mat overlap_matrix() {
        const int N = basis_count();
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

    arma::cube overlap_matrix_grad() {
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
                for (int j = 0; j < S_i.n_cols; j++) {
                    S.insert_cols(S.n_cols, S_i.col(j));
                }
            }
        }
        return S;
    }

    arma::mat hamiltonian(const double K = 1.75) {
        const int N = basis_count();
        arma::mat S = overlap_matrix();
        arma::mat H = arma::zeros(N, N);

        // Set diagonals to ionization potentials
        int i = 0;
        for (Atom atom : atoms) {
            for (CGAO orbital : atom.orbitals) {
                H(i, i) = orbital.h;
                i++;
            }
        }

        // Compute off-diagonal elements
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                const double H_ij = 0.5 * K * (H(i, i) + H(j, j)) * S(i, j);
                H(i, j) = H_ij;
                H(j, i) = H_ij;
            }
        }

        return H;
    }

    double nuclear_repulsion_energy() {
        double energy = 0.0;
        for (int a = 0; a < atoms.size(); a++) {
            Atom A = atoms[a];
            for (int b = 0; b < a; b++) {
                Atom B = atoms[b];
                energy += 27.211324570273 * A.Z * B.Z / arma::norm(A.r - B.r);
            }
        }
        return energy;
    }

    arma::vec nuclear_repulsion_energy_grad(int a) {
        Atom A = atoms[a];
        arma::vec result = arma::zeros(3);
        for (int b = 0; b < atoms.size(); b++) {
            Atom B = atoms[b];
            if (b != a) {
                result -= 27.211324570273 * A.Z * B.Z * (A.r - B.r) / pow(arma::norm(A.r - B.r), 3);
            }
        }
        return result;
    }

    double electronic_energy() {
        arma::mat Pa = density(Calpha, n_alpha);
        arma::mat Pb = density(Cbeta, n_beta);
        arma::mat H = core_hamiltonian();
        arma::mat Fa = fock_matrix(true);
        arma::mat Fb = fock_matrix(false);
        return 0.5 * arma::accu(Pa % (H + Fa)) + 0.5 * arma::accu(Pb % (H + Fb));
    }

    arma::vec electronic_energy_grad(int a) {
        Atom A = atoms[a];
        arma::vec result = arma::zeros(3);
        arma::mat Pa = density(Calpha, n_alpha);
        arma::mat Pb = density(Cbeta, n_beta);
        arma::mat P = Pa + Pb;
        arma::vec Ptot = total_density_matrix();
        arma::cube S_grad = overlap_matrix_grad();
        int j = 0;
        for (int b = 0; b < atoms.size(); b++) {
            Atom B = atoms[b];
            arma::vec gammaAB_RA = eri_grad(A, B);
            result += (Ptot(a) * Ptot(b) - B.Z * Ptot(a) - A.Z * Ptot(b)) * gammaAB_RA;
            for (CGAO Borbital : B.orbitals) {
                int i = 0;
                for (int c = 0; c < atoms.size(); c++) {
                    Atom C = atoms[c];
                    for (CGAO Corbital : C.orbitals) {
                        if ((c == a) && (a != b)) {
                            result -= (pow(Pa(i, j), 2) + pow(Pb(i, j), 2)) * gammaAB_RA;
                            result -= (Corbital.beta + Borbital.beta) * P(i, j) * overlap_grad(Corbital, Borbital);
                        }
                        i++;
                    }
                }
                j++;
            }
        }
        return result;
    }

    double energy(arma::vec epsilon) {
        double energy = 0.0;
        for (int i = 0; i < electron_count(); i++) {
            energy += 2*epsilon(i);
        }
        return energy;
    }

    double energy() {
        if (kind == "CNDO2") {
            return electronic_energy() + nuclear_repulsion_energy();
        } else if (kind == "SD_with_line_search" || kind == "standard_SD") {
            return lennard_jones_energy();
        } else {
            throw std::invalid_argument("Invalid kind of simulation!");
        }
    }

    double lennard_jones_energy() {
        double energy = 0.0;
        for (int i = 0; i < atoms.size(); i++) {
            for (int j = i + 1; j < atoms.size(); j++) {
                double R_ij = arma::norm(atoms[i].r - atoms[j].r);
                energy += epsilon * (pow(sigma / R_ij, 12) - 2 * pow(sigma / R_ij, 6));
            }
        }
        return energy;
    }
    arma::mat lennard_jones_numerical_force(const double h, const bool center) {
        arma::mat forces = arma::zeros(3, atoms.size());
        double energy = lennard_jones_energy();
        for (int i = 0; i < atoms.size(); i++) {
            for (int k = 0; k < 3; k++) {
                double energy_left = energy;
                double energy_right = energy;
                for (int j = 0; j < atoms.size(); j++) {
                    if (i == j) {
                        continue;
                    }
                    double R_ij = norm(atoms[i].r - atoms[j].r);
                    double energy_contribution = epsilon * (pow(sigma / R_ij, 12) - 2 * pow(sigma / R_ij, 6));
                    energy_left -= energy_contribution;
                    energy_right -= energy_contribution;
                    arma::vec atom_left = atoms[i].r;
                    atom_left[k] -= h;
                    arma::vec atom_right = atoms[i].r;
                    atom_right[k] += h;
                    R_ij = norm(atom_left - atoms[j].r);
                    energy_left += epsilon * (pow(sigma / R_ij, 12) - 2 * pow(sigma / R_ij, 6));
                    R_ij = norm(atom_right - atoms[j].r);
                    energy_right += epsilon * (pow(sigma / R_ij, 12) - 2 * pow(sigma / R_ij, 6));
                }
                if (center) {
                    forces(k, i) = -(energy_right - energy_left) / (2.0 * h);
                }
                else {
                    forces(k, i) = -(energy_right - energy) / h;
                }
            }
        }
        return forces;
    }
    arma::mat lennard_jones_analytic_force() {
        arma::mat forces = arma::zeros(3, atoms.size());
        for (int i = 0; i < atoms.size(); i++) {
            for (int j = 0; j < atoms.size(); j++) {
                if (i == j) {
                    continue;
                }
                double R_ij = norm(atoms[i].r - atoms[j].r);
                double mag = 12 * epsilon * (pow(sigma / R_ij, 12) - pow(sigma / R_ij, 6)) / (R_ij * R_ij);
                forces.col(i) += mag * (atoms[i].r - atoms[j].r);
            }
        }
        return forces;
    }
    struct bracket {
        double a;
        double b;
        double c;
    };
    bracket bracket_minimum(double a, double b, const double GOLDEN_RATIO = 1.618304) {
        auto f = [&] (double step) -> double {
            Simulation temp = Simulation(*this, forces, step);
            return temp.energy();
        };
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
                }
                else if (fu > fb) {
                    c = u;
                    fc = fu;
                    break;
                }
                u = c + GOLDEN_RATIO * (c - b);
            }
            else if ((c - u) * (u - ulim) > 0.0) {
                if (fu < fc) {
                    fu = f(u);
                    b = c;
                    fb = fc;
                    c = u;
                    fc = fu;
                    u = u + GOLDEN_RATIO * (u - c);
                }
            }
            else {
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
    Simulation line_search(std::ostream &fout, double a, double b, const double GOLDEN_RATIO = 1.618304, const double TOL = 3.0e-8, const double h = 1e-4, const double l = 0.3) {
        double xmin = 0.0;
        auto f = [&](double step) -> double {
            Simulation temp = Simulation(*this, forces, step);
            return temp.energy();
        };

        if ((kind == "CNDO2") || (kind == "SD_with_line_search")) {
            fout << "Start golden section search" << std::endl;
        }
        int flag = 0;
        bracket results;
        if ((kind == "CNDO2") || (kind == "SD_with_line_search")) {
            results = bracket_minimum(a, b, GOLDEN_RATIO);
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
            }
            else {
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
                }
                else {
                    x0 = x1;
                    x1 = x2;
                    x2 = x2 / GOLDEN_RATIO + x3 * (1.0 - 1.0 / GOLDEN_RATIO);
                    f1 = f2;
                    f2 = f(x2);
                }
            }
            if (f1 < f2) {
                xmin = x1;
            }
            else {
                xmin = x2;
            }
        }
        else {
            // Do steepest descent
            xmin = l;
        }
        Simulation sim = Simulation(*this, forces, xmin);
        if (sim.energy() > this->energy()) {
            return Simulation(*this);
        }
        else {
            return sim;
        }
    }
};


std::ostream& operator << (std::ostream& os, const Simulation& sim) {
    return (os << sim.atoms);
}