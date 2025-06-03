#pragma once

#include <armadillo>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "simulation.hpp"

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