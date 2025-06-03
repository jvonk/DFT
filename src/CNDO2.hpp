#pragma once

#include <armadillo>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "simulation.hpp"

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

    arma::mat total_density_matrix() const
    {
        int N = basis_count();
        arma::mat P = arma::zeros(N, N);
        if (Calpha.n_cols > 0) {
            P += density_matrix(Calpha);
        }
        if (Cbeta.n_cols > 0) {
            P += density_matrix(Cbeta);
        }
        return P;
    }

    arma::vec total_density_vector() const
    {
        arma::mat P = total_density_matrix();
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
        arma::vec Ptot = total_density_vector();
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
        arma::vec Ptot = total_density_vector();
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