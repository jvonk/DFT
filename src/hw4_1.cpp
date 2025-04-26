#include <armadillo> 
#include <iostream>
#include <string> 
#include <iomanip>

#include "header_file.hpp"


int main(int argc, char** argv){
    std::cout << "Homework 4!" << std::endl;
    std::string input_path = std::string(argv[1]);
    std::cout << "Input Path: " << input_path << std::endl;
    std::ifstream fin(input_path);
    const double tol = 1e-6;
    int n;
    int n_alpha;
    int n_beta;
    fin >> n >> n_alpha >> n_beta;
    std::cout << "Number of atoms: " << n << std::endl;
    std::vector<Atom> atoms;
    for (int i = 0; i < n; i++) {
        int E;
        double X, Y, Z;
        fin >> E >> X >> Y >> Z;
        arma::vec R = { X, Y, Z };
        // std::cout << E << X << Y << Z << std::endl;
        Atom atom = { E, R };
        if ((E != 1) && (E != 6) && (E != 7) && (E != 8) && (E != 9)) {
            throw std::invalid_argument("Element not supported!");
        }
        atoms.push_back(atom);
    }
    Simulation sim(atoms, "CNDO2", n_alpha, n_beta);
    std::cout << "gamma" << std::endl;
    arma::mat gamma = arma::zeros(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            gamma(i, j) = eri(atoms[i], atoms[j]);
        }
    }
    std::cout << gamma;
    std::cout << "Overlap" << std::endl;
    std::cout << sim.overlap_matrix();
    std::cout << " p =  " << n_alpha << " q =  " << n_beta << std::endl;
    std::cout << "H_core" << std::endl;
    arma::mat H = sim.core_hamiltonian();
    std::cout << H;
    const int N = sim.basis_count();
    arma::mat Falpha = arma::zeros(N, N);
    arma::mat Fbeta = arma::zeros(N, N);
    for (int iteration = 0; ; iteration++) {
        std::cout << "Iteration: " << iteration << std::endl;
        std::cout << "Fa" << std::endl;
        arma::mat Fa = sim.fock_matrix(true);
        std::cout << Fa;
        std::cout << "Fb" << std::endl;
        arma::mat Fb = sim.fock_matrix(false);
        std::cout << Fb;
        std::cout << "after solving eigen equation: " << iteration << std::endl;
        std::cout << "Ca" << std::endl;
        arma::vec epsilona;
        arma::mat Ca;
        arma::eig_sym(epsilona, Ca, Fa);
        std::cout << Ca;
        std::cout << "Cb" << std::endl;
        arma::vec epsilonb;
        arma::mat Cb;
        arma::eig_sym(epsilonb, Cb, Fb);
        std::cout << Cb;
        std::cout << " p =  " << n_alpha << " q =  " << n_beta << std::endl;
        arma::mat Pa_new = density(Ca, sim.n_alpha);
        std::cout << Pa_new;
        arma::mat Pb_new = density(Cb, sim.n_beta);
        arma::mat Pa = density(sim.Calpha, sim.n_alpha);
        arma::mat Pb = density(sim.Cbeta, sim.n_beta);
        sim.Calpha = Ca;
        sim.Cbeta = Cb;
        if ((arma::norm(Pa_new - Pa, "inf") < tol) || (arma::norm(Pb_new - Pb, "inf") < tol)) {
            std::cout << "Ea" << std::endl;
            std::cout << epsilona;
            std::cout << "Eb" << std::endl;
            std::cout << epsilonb;
            std::cout << "Ca" << std::endl;
            std::cout << Ca;
            std::cout << "Cb" << std::endl;
            std::cout << Cb;
            break;
        }
        std::cout << "Pa_new" << std::endl;
        std::cout << Pa_new;
        std::cout << "Pb_new" << std::endl;
        std::cout << Pb_new;
        std::cout << "P_t" << std::endl;
        int i = 0;
        arma::vec Ptot = arma::zeros(sim.atoms.size());
        for (int a = 0; a < sim.atoms.size(); a++) {
            Atom A = sim.atoms[a];
            for (CGAO Aorbital : A.orbitals) {
                Ptot(a) += Pa_new(i, i);
                Ptot(a) += Pb_new(i, i);
                i++;
            }
        }
        std::cout << Ptot;
        std::cout << "Ga" << std::endl;
        std::cout << "Gb" << std::endl;
    }
    double E_nuc = sim.nuclear_repulsion_energy();
    std::cout << "Nuclear Repulsion Energy is " << E_nuc << " eV." << std::endl;
    double E_elec = sim.electronic_energy();
    std::cout << "Electron Energy is " << E_elec << " eV." << std::endl;
    double E_total = E_nuc + E_elec;
    std::cout << "The molecule in file " << input_path << " has energy " << E_total << std::endl;
}