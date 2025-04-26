#include <armadillo> 
#include <iostream>
#include <iomanip>

#include "header_file.hpp"

int main(int argc, char** argv){
    std::cout << "Question 2" << std::endl; 
    Shell shellA, shellB;
    std::cin >> shellA;
    std::cout << "Shell 1 has " << shellA.n_functions() << " functions." << std::endl;
    std::cout << shellA << std::endl;
    std::cin >> shellB;
    std::cout << "Shell 2 has " << shellB.n_functions() << " functions." << std::endl;
    std::cout << shellB << std::endl;

    arma::mat result = gaussian_overlap(shellA, shellB);
    std::cout << "Overlap integral between Shell 1 and Shell 2" << std::endl;
    std::cout << result;
    // Overlap integral between Shell 1 and Shell 2
    //         0        0        0
    // The components of angular momentum (l, m, n) for the matrix column, from top to bottom, are listed sequentially as: (0, 0, 0).
    // The components of angular momentum (l, m, n) for the matrix row, from left to right, are listed sequentially as: (1, 0, 0), (0, 1, 0), (0, 0, 1).
    std::cout << "The components of angular momentum (l, m, n) for the matrix column, from top to bottom, are listed sequentially as: ";
    arma::Mat<int> lA = shellA.lmat();
    for (int i = 0; i < lA.n_rows; i++) {
        std::cout << "(" << lA(i, 0) << ", " << lA(i, 1) << ", " << lA(i, 2) << ")";
        if (i < lA.n_rows - 1) {
            std::cout << ", ";
        } else {
            std::cout << "." << std::endl;
        }
    }
    arma::Mat<int> lB = shellB.lmat();
    std::cout << "The components of angular momentum (l, m, n) for the matrix row, from left to right, are listed sequentially as: ";
    for (int i = 0; i < lB.n_rows; i++) {
        std::cout << "(" << lB(i, 0) << ", " << lB(i, 1) << ", " << lB(i, 2) << ")";
        if (i < lB.n_rows - 1) {
            std::cout << ", ";
        } else {
            std::cout << "." << std::endl;
        }
    }
    return 0;
}