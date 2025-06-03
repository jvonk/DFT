#pragma once

#include <armadillo>
#include <cmath>
#include <iomanip>
#include <iostream>

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
    arma::uword n_alpha;
    arma::uword n_beta;
    double L;
    double E_cutoff;
    arma::uword N_grid;
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

    // Cached data
    

    DFT(const std::vector<Atom> &atoms, const arma::uword N_ALPHA, const arma::uword N_BETA,
        const double BOX_SIZE_ANGSTROM, const double KINETIC_ENERGY_CUTOFF_EV,
        const arma::uword NUMBER_GRID_POINTS, const int INCLUDE = 3,
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
            arma::sp_mat Fa_sparse = arma::sp_mat(Fa);
            arma::sp_mat Fb_sparse = arma::sp_mat(Fb);
            arma::vec epsilona;
            arma::mat Ca;
            // arma::eig_sym(epsilona, Ca, Fa);
            arma::eigs_sym(epsilona, Ca, Fa_sparse, n_alpha, "sa");
            // Ca = Ca.head_cols(n_alpha);
            arma::vec epsilonb;
            arma::mat Cb;
            // arma::eig_sym(epsilonb, Cb, Fb);
            arma::eigs_sym(epsilonb, Cb, Fb_sparse, n_beta, "sa");
            // Cb = Cb.head_cols(n_beta);
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
                if (n_alpha > 0) {
                    std::cout << "Energy of occupied orbital (alpha) 0: "
                            << epsilona(0) << std::endl;
                }
                std::cout << "Kinetic energy: " << kinetic_energy()
                          << std::endl;
                std::cout << "Hartree energy: " << hartree_energy()
                          << std::endl;
                std::cout << "External energy: " << external_energy()
                          << std::endl;
                std::cout << "Exchange-correlation energy (alpha): "
                          << exchange_correlation_energy(Calpha) << std::endl;
                if (n_beta > 0) {
                    std::cout << "Energy of occupied orbital (beta) 0: "
                              << epsilonb(0) << std::endl;
                }
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
            if ((n_alpha == 0 || arma::norm(Pa_new - Pa, "inf") < tol) && (n_beta == 0 || arma::norm(Pb_new - Pb, "inf") < tol)) {
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
        if (n_alpha > 0) {
            rho += density(Calpha);
        }
        if (n_beta > 0) {
            rho += density(Cbeta);
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
    
    arma::mat total_density_matrix() const
    {
        arma::mat P = arma::zeros(basis.size(), basis.size());
        if (Calpha.n_cols > 0) {
            P += density_matrix(Calpha);
        }
        if (Cbeta.n_cols > 0) {
            P += density_matrix(Cbeta);
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