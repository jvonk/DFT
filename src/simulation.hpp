#pragma once

#include <armadillo>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "GAO.hpp"

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