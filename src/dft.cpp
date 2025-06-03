#include <iostream> 
#include <fstream>
#include <filesystem>
#include <string> 
#include <cstdlib>
#include <stdexcept>


#include <nlohmann/json.hpp> // This is the JSON handling library
#include <armadillo> 

#include "dft.hpp"

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
    std::cout << "Config file: " << argv[1] << std::endl;
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

    double BOX_SIZE_ANGSTROM = config["box_size_angstrom"]; 
    double KINETIC_ENERGY_CUTOFF_EV = config["kinetic_energy_cutoff_eV"];
    arma::uword NUMBER_GRID_POINTS = config["number_grid_points"]; 

    arma::uword N_ALPHA = config["num_alpha_electrons"];
    arma::uword N_BETA = config["num_beta_electrons"];

    // int INCLUDE = config["included_terms"];
    int INCLUDE = 3;
    // bool DENSITY_FITTING = config["density_fitting"];
    bool DENSITY_FITTING = false;
    // double TOL = config["tolerance"];
    double TOL = 1e-6;

    std::ifstream atoms_file(atoms_file_path);
    std::cout << "Atoms file: " << atoms_file_path << std::endl;
    arma::uword num_3D_dims = 3; 
    arma::uword num_atoms;
    atoms_file >> num_atoms >> std::ws;
    std::cout << "Number of atoms: " << num_atoms << std::endl;
    std::string comment;
    std::getline(atoms_file, comment);
    std::cout << "Comment: " << comment << std::endl;
    std::vector<Atom> atoms;
    for (arma::uword i = 0; i < num_atoms; i++) {
        int E;
        double X, Y, Z;
        atoms_file >> E >> X >> Y >> Z;
        arma::vec R = { X, Y, Z };
        Atom atom(E, R, false);
        std::cout << "Atom: " << atom << std::endl;
        atoms.push_back(atom);
    }
    DFT sim(atoms, N_ALPHA, N_BETA, BOX_SIZE_ANGSTROM, KINETIC_ENERGY_CUTOFF_EV, NUMBER_GRID_POINTS, INCLUDE, DENSITY_FITTING, TOL);
    sim.converge(true);
    std::cout << std::fixed << std::setprecision(4) << std::setw(8) << std::right;
    std::cout << sim.energy() << " eV" << std::endl;

    // check that output dir exists
    if (!fs::exists(output_file_path.parent_path())){
        fs::create_directories(output_file_path.parent_path()); 
    }
    
    // delete the file if it does exist (so that no old answers stay there by accident)
    if (fs::exists(output_file_path)){
        fs::remove(output_file_path); 
    }
    sim.Calpha.save(arma::hdf5_name(output_file_path, "C_alpha", arma::hdf5_opts::append));
    sim.Cbeta.save(arma::hdf5_name(output_file_path, "C_beta", arma::hdf5_opts::append));
    sim.grid_points.save(arma::hdf5_name(output_file_path, "grid_points", arma::hdf5_opts::append));
    sim.grid_wavefunction.save(arma::hdf5_name(output_file_path, "grid_wavefunction", arma::hdf5_opts::append));
    sim.density(sim.Calpha).save(arma::hdf5_name(output_file_path, "density_alpha", arma::hdf5_opts::append));
    sim.density(sim.Cbeta).save(arma::hdf5_name(output_file_path, "density_beta", arma::hdf5_opts::append));
}  
