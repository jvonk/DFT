#include <iostream> 
#include <fstream>
#include <filesystem>
#include <string> 
#include <cstdlib>
#include <stdexcept>

// RENAME THIS FILE TO hw5_2 IF YOU ARE GOING TO USE IT 

#include <nlohmann/json.hpp> // This is the JSON handling library
#include <armadillo> 

#include "header_file.hpp"

// convenience definitions so the code is more readable
namespace fs = std::filesystem;
using json = nlohmann::json; 

int main(int argc, char** argv){
    // check that a config file is supplied
    if (argc != 2){
        std::cerr << "Usage: " << argv[0] << " path/to/config.json" << std::endl; 
        return EXIT_FAILURE; 
    }
    
    // parse the config file 
    fs::path config_file_path(argv[1]);
    if (!fs::exists(config_file_path)){
        std::cerr << "Path: " << config_file_path << " does not exist" << std::endl; 
        return EXIT_FAILURE;
    }

    std::ifstream config_file(config_file_path); 
    json config = json::parse(config_file); 

    // extract the important info from the config file
    fs::path atoms_file_path = config["atoms_file_path"];
    fs::path output_file_path = config["output_file_path"];  
    int num_alpha_electrons = config["num_alpha_electrons"];
    int num_beta_electrons = config["num_beta_electrons"];

    // Your answers go in these objects 
    std::ifstream atoms_file(atoms_file_path);
    int num_3D_dims = 3; 
    int num_atoms;
    atoms_file >> num_atoms;
    std::cout << "Number of atoms: " << num_atoms << std::endl;
    std::string comment;
    std::getline(atoms_file, comment);
    std::getline(atoms_file, comment);
    std::cout << "Comment: " << comment << std::endl;
    std::vector<Atom> atoms;
    for (int i = 0; i < num_atoms; i++) {
        int E;
        double X, Y, Z;
        atoms_file >> E >> X >> Y >> Z;
        arma::vec R = { X, Y, Z };
        Atom atom = { E, R };
        std::cout << "Atom: " << atom << std::endl;
        if ((E != 1) && (E != 6) && (E != 7) && (E != 8) && (E != 9)) {
            throw std::invalid_argument("Element not supported!");
        }
        atoms.push_back(atom);
    }
    Simulation sim(atoms, "CNDO2", num_alpha_electrons, num_beta_electrons);
    int num_basis_functions = sim.basis_count();
    
    arma::mat Suv_RA(num_3D_dims, num_basis_functions*num_basis_functions); 
    // Ideally, this would be (3, n_funcs, n_funcs) rank-3 tensor
    // but we're flattening (n-funcs, n-atoms) into a single dimension (n-funcs ^ 2)
    // this is because tensors are not supported in Eigen and I want students to be able to 
    // submit their work in a consistent format
    arma::mat gammaAB_RA(num_3D_dims, num_atoms*num_atoms); 
    // This is the same story, ideally, this would be (3, num_atoms, num_atoms) instead of (3, num_atoms ^ 2)
    arma::mat gradient_nuclear(num_3D_dims, num_atoms);
    arma::mat gradient_electronic(num_3D_dims, num_atoms); 
    arma::mat gradient(num_3D_dims, num_atoms); 

    // most of the code goes here
    // Set print configs
    std::cout << std::fixed << std::setprecision(4) << std::setw(8) << std::right; 

    double force_threshold = 1e-3;
    const double tol = 1e-10;
    const double golden_ratio = 1.618304;
    double h = 0.0001;
    double l = 0.01;
    for (int iteration = 0; ; iteration++) {
        arma::mat Fa = sim.fock_matrix(true);
        arma::mat Fb = sim.fock_matrix(false);
        arma::vec epsilona;
        arma::mat Ca;
        arma::eig_sym(epsilona, Ca, Fa);
        arma::vec epsilonb;
        arma::mat Cb;
        arma::eig_sym(epsilonb, Cb, Fb);
        arma::mat Pa_new = density(Ca, sim.n_alpha);
        arma::mat Pb_new = density(Cb, sim.n_beta);
        arma::mat Pa = density(sim.Calpha, sim.n_alpha);
        arma::mat Pb = density(sim.Cbeta, sim.n_beta);
        sim.Calpha = Ca;
        sim.Cbeta = Cb;
        if ((arma::norm(Pa_new - Pa, "inf") < tol) || (arma::norm(Pb_new - Pb, "inf") < tol)) {
            break;
        }
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
    }

    arma::cube Suv_grad = sim.overlap_matrix_grad();
    for (int dir = 0; dir < 3; dir++) {
        Suv_RA.row(dir) = Suv_grad.row_as_mat(dir).as_row();
    }

    for (int a = 0; a < num_atoms; a++) {
        for (int b = 0; b < num_atoms; b++) {
            gammaAB_RA.col(a * num_atoms + b) = eri_grad(atoms[a], atoms[b]);
        }
        gradient_nuclear.col(a) = sim.nuclear_repulsion_energy_grad(a);
        gradient_electronic.col(a) = sim.electronic_energy_grad(a);
        gradient.col(a) = gradient_nuclear.col(a) + gradient_electronic.col(a);
    }


    // You do not need to modify the code below this point 

    // Set print configs
    std::cout << std::fixed << std::setprecision(4) << std::setw(8) << std::right; 
    std::cout << "Nuclear Repulsion Energy is " << sim.nuclear_repulsion_energy() << " eV." << std::endl;
    std::cout << "Electron Energy is " << sim.electronic_energy() << " eV." << std::endl;


    // inspect your answer via printing
    std::cout << std::fixed << std::setprecision(4) << std::setw(8) << std::right; 
    Suv_RA.print("Suv_RA");
    gammaAB_RA.print("gammaAB_RA");
    gradient_nuclear.print("gradient_nuclear");
    gradient_electronic.print("gradient_electronic");
    gradient.print("gradient");

    // check that output dir exists
    if (!fs::exists(output_file_path.parent_path())){
        fs::create_directories(output_file_path.parent_path()); 
    }
    
    // delete the file if it does exist (so that no old answers stay there by accident)
    if (fs::exists(output_file_path)){
        fs::remove(output_file_path); 
    }

    // write results to file 
    Suv_RA.save(arma::hdf5_name(output_file_path, "Suv_RA", arma::hdf5_opts::append));
    gammaAB_RA.save(arma::hdf5_name(output_file_path, "gammaAB_RA", arma::hdf5_opts::append));
    gradient_nuclear.save(arma::hdf5_name(output_file_path, "gradient_nuclear", arma::hdf5_opts::append));
    gradient_electronic.save(arma::hdf5_name(output_file_path, "gradient_electronic", arma::hdf5_opts::append));
    gradient.save(arma::hdf5_name(output_file_path, "gradient", arma::hdf5_opts::append));



    // Extra Credit Portion
    sim.forces = -gradient;
    int iteration = 1;
    double energy = sim.energy();
    for (; iteration < 1000; iteration++) {
        std::cout << "Iteration: " << iteration << std::endl;
        double old_energy = energy;
        sim = sim.line_search(std::cout, 0.0, l, golden_ratio, 3.0e-8, h, l);
        for (int a = 0; a < num_atoms; a++) {
            gradient.col(a) = sim.nuclear_repulsion_energy_grad(a) + sim.electronic_energy_grad(a);
        }
        sim.forces = -gradient;
        energy = sim.energy();
        std::cout << "new_point" << std::endl;;
        for (Atom atom : sim.atoms) {
            std::cout << atom << std::endl;
        }
        std::cout << "current energy: " << energy << std::endl;
        std::cout << "force:" << std::endl;
        std::cout << sim.forces;
        if ((norm2(sim.forces) < force_threshold) || (abs(energy - old_energy) < tol)) {
            break;
        }
        if (energy < old_energy) {
            l *= 1.01;
        }
        else {
            l /= 2.0;
        }
    }
    std::cout << "Total iterations: " << iteration << std::endl;
    energy = sim.energy();
    std::cout << "Final energy: " << energy << std::endl;
    std::cout << "Optimized structure:" << std::endl;
    for (Atom atom : sim.atoms) {
        std::cout << atom << std::endl;
    }
}  