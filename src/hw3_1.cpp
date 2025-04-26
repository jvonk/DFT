#include <armadillo> 
#include <iostream>
#include <string> 
#include <iomanip>

#include "header_file.hpp"

int main(int argc, char** argv){
    std::cout << "Homework 3!" << std::endl;
    std::string input_path = std::string(argv[1]);
    std::cout << "Input Path: " << input_path << std::endl;
    std::ifstream fin(input_path);
    int n;
    int unknown;
    fin >> n >> unknown;
    std::cout << "Number of atoms: " << n << std::endl;
    std::vector<Atom> atoms;
    for (int i = 0; i < n; i++) {
        int E;
        double X, Y, Z;
        fin >> E >> X >> Y >> Z;
        arma::vec R = { X, Y, Z };
        Atom atom = { E, R };
        if ((E != 1) && (E != 6)) {
            throw std::invalid_argument("Not hydrogen or carbon!");
        }
        atoms.push_back(atom);
    }
    Simulation sim(atoms);
    
    std::cout << "Overlap matrix:" << std::endl;
    arma::mat S = sim.overlap_matrix();
    std::cout << S;
    std::cout << "Hamiltonian matrix:" << std::endl;
    arma::mat H = sim.hamiltonian();
    std::cout << H;
    std::cout << "X_mat:" << std::endl;
    arma::mat X = arma::inv(arma::sqrtmat_sympd(S));\
    std::cout << X;
    std::cout << "MO coefficients (C matrix):" << std::endl;
    arma::mat Hdiag = X.t() * H * X;
    arma::vec epsilon;
    arma::mat V;
    arma::eig_sym(epsilon, V, Hdiag);
    arma::mat C = X * V;
    std::cout << C;
    std::cout << "MO overlap matrix:" << std::endl;
    std::cout << C.t() * S * C;
    double energy = sim.energy(epsilon);
    std::cout << std::fixed << std::setprecision(6) << "The molecule in file " << input_path << " has energy " << energy << std::endl;
    return 0;
}