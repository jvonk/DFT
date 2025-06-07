#pragma once

#include <armadillo>
#include <cmath>
#include <iomanip>
#include <iostream>
extern "C" {
    #include <xc.h>
}

#include "simulation.hpp"
#include "tools.hpp"

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
    arma::uword na, nb;
    double L;
    double E_cutoff;
    arma::uword N_grid;
    arma::mat grid_points;
    arma::mat Ca, Cb;
    arma::mat Pa, Pb;
    double N;
    std::vector<PB> basis;
    std::vector<PB> auxilliary_basis;
    arma::mat grid_wavefunction;
    arma::mat grid_auxilliary_wavefunction;
    double grid_weight;
    int include;
    bool density_fitting;
    double tol;
    bool functional;
    xc_func_type func;

    DFT(const std::vector<Atom> &atoms, const arma::uword NA, const arma::uword NB,
        const double BOX_SIZE_ANGSTROM, const double KINETIC_ENERGY_CUTOFF_EV,
        const arma::uword NUMBER_GRID_POINTS, const int INCLUDE = 3,
        const bool DENSITY_FITTING = true,
        const double TOL = std::numeric_limits<double>::epsilon(),
        const int FUNCTIONAL = true)
        : Simulation(atoms), na(NA), nb(NB),
          L(BOX_SIZE_ANGSTROM / 0.529177210544),
          E_cutoff(KINETIC_ENERGY_CUTOFF_EV / 27.211386246),
          N_grid(NUMBER_GRID_POINTS), include(INCLUDE),
          density_fitting(DENSITY_FITTING), tol(TOL), functional(FUNCTIONAL)
    {
        if (xc_func_init(&func, functional, XC_UNPOLARIZED) != 0) {
            throw std::invalid_argument("Functional not found");
        }
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
                        auxilliary_basis.push_back(PB(arma::uvec({i, j, k}), L));
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

        Ca = arma::zeros(basis.size(), na);
        Cb = arma::zeros(basis.size(), nb);

        Pa = arma::zeros(basis.size(), basis.size());
        Pb = arma::zeros(basis.size(), basis.size());

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
        std::cout << "Number of electrons: " << na + nb << std::endl;
        std::cout << "Real space density initialized (zeros)" << std::endl;
        std::cout << "Will finish constructing the kinetic energy part of the "
                     "Hamiltonian later"
                  << std::endl;
        std::cout << "Finished evaluating the basis set at each grid point"
                  << std::endl;
    }

    void scf(const bool debug = false)
    {
        for (int iteration = 0;; iteration++) {
            bool converged = true;

            if (na > 0) {
                arma::mat Fa = fock_matrix(Ca);
                arma::sp_mat Fa_sparse = arma::sp_mat(Fa);
                arma::vec epsilona;
                arma::mat Ca_new;
                arma::eigs_sym(epsilona, Ca_new, Fa_sparse, na, "sa");
                Ca = Ca_new;
                arma::mat Pa_old = Pa;
                Pa = density_matrix(Ca);
                converged &= arma::norm(Pa - Pa_old, "inf") < tol;
            }

            if (nb > 0) {
                arma::mat Fb = fock_matrix(Cb);
                arma::sp_mat Fb_sparse = arma::sp_mat(Fb);
                arma::vec epsilonb;
                arma::mat Cb_new;
                arma::eigs_sym(epsilonb, Cb_new, Fb_sparse, nb, "sa");
                Cb = Cb_new;
                arma::mat Pb_old = Pb;
                Pb = density_matrix(Cb);
                converged &= arma::norm(Pb - Pb_old, "inf") < tol;
            }

            if (converged) {
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

    arma::vec total_density() const
    {
        arma::vec rho = arma::zeros(grid_points.n_cols);
        if (na > 0) {
            rho += density(Ca);
        }
        if (nb > 0) {
            rho += density(Cb);
        }
        return rho;
    }

    arma::mat external_potential_matrix() const
    {
        arma::vec Vext = arma::zeros(grid_points.n_cols);
        if (density_fitting) {
            arma::vec rho = total_density();
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
        arma::vec rho = total_density();
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
        if (C.n_cols == 0) {
            return arma::zeros(basis.size(), basis.size());
        }
        arma::vec Vxc = arma::zeros(grid_points.n_cols);
        arma::vec rho = density(C);
        if (functional > 0) {
            xc_lda_vxc(&func, grid_points.n_cols, rho.memptr(), Vxc.memptr());
        } else {
            Vxc = -arma::pow((3.0 / M_PI) * rho, 1.0 / 3);
        }
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
    
    arma::mat total_density_matrix() const
    {
        arma::mat P = arma::zeros(basis.size(), basis.size());
        if (Ca.n_cols > 0) {
            P += density_matrix(Ca);
        }
        if (Cb.n_cols > 0) {
            P += density_matrix(Cb);
        }
        return P;
    }

    double kinetic_energy() const
    {
        arma::mat P = total_density_matrix();
        arma::mat T = kinetic_matrix();
        return arma::trace(P * T);
    }

    double hartree_energy() const
    {
        arma::mat P = total_density_matrix();
        arma::mat VH = hartree_potential_matrix();
        return 0.5 * arma::trace(P * VH);
    }

    double external_energy() const
    {
        arma::mat P = total_density_matrix();
        arma::mat Vext = external_potential_matrix();
        return arma::trace(P * Vext);
    }

    double exchange_correlation_energy(const arma::mat &C) const
    {
        if (C.n_cols == 0) {
            return 0.0;
        }
        arma::vec Exc = arma::zeros(grid_points.n_cols);
        arma::vec rho = density(C);
        if (functional > 0) {
            xc_lda_exc(&func, grid_points.n_cols, rho.memptr(), Exc.memptr());
            Exc %= rho;
        } else {
            Exc = -0.75 * pow(3.0 / M_PI, 1.0 / 3) * arma::pow(rho, 4.0 / 3);
        }
        return grid_weight * arma::accu(Exc);
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
            E += exchange_correlation_energy(Ca) +
                 exchange_correlation_energy(Cb);
        }
        return E;
    }
};