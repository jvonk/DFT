#include <iostream> 
#include <fstream>
#include <filesystem>
#include <string> 
#include <cstdlib>
#include <stdexcept>


#include <nlohmann/json.hpp> // This is the JSON handling library

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

    double box_size_angstrom = config['box_size_angstrom']; 
    double kinetic_energy_cutoff_eV = config['kinetic_energy_cutoff_eV'];
    int number_grid_points = config['number_grid_points']; 

    int num_alpha_electrons = config["num_alpha_electrons"];
    int num_beta_electrons = config["num_beta_electrons"];
    
}  
