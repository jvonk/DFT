#pragma once

#include <armadillo>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "simulation.hpp"

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