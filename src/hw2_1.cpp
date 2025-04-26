#include <armadillo> 
#include <iostream>
#include <iomanip>

#include "header_file.hpp"
int main(int argc, char** argv){
    std::cout << "Question 1" << std::endl;
    const double tol = 1e-20;
    double XA, XB, alpha, beta;
    int lA, lB;
    std::cin >> XA >> alpha >> lA;
    std::cin >> XB >> beta >> lB;
    std::cout << "Gaussian 1: " << XA << " " << alpha << " " << lA << std::endl;
    std::cout << "Gaussian 2: " << XB << " " << beta << " " << lB << std::endl;
    const double width = sqrt(400.0 / (alpha + beta)) * (1.0 + lA + lB);
    std::cout << "Width: " << width << std::endl;
    const double left = (XA + XB - width)/2.0;
    const double right = (XA + XB + width)/2.0;
    std::cout << "Left: " << left << std::endl;
    std::cout << "Right: " << right << std::endl;
    DoubleGaussian orbitals(XA, lA, alpha, XB, lB, beta);
    Trapezoid trapezoid(orbitals, left, right);
    std::cout << std::setprecision(17) << std::scientific;
    std::cout << "1d numerical overlap integral between Gaussian functions is " << trapezoid(tol) << std::endl;
    std::cout << std::defaultfloat;
    
    // Testing:
    // double offsets[] = {0.0, 1.0};
    // for (double offset : offsets) {
    //     std::cout << "At offset: " << offset << std::endl;
    //     DoubleGaussian ss_origin(-offset/2, 0, 0.5, offset/2, 0, 0.5);
    //     Trapezoid trapezoid_ss(ss_origin, -lim, lim);
    //     std::cout << "Two s-type functions: " << trapezoid_ss(tol) << std::endl;
    //     DoubleGaussian sp_origin(-offset/2, 0, 0.5, offset/2, 1, 0.5);
    //     Trapezoid trapezoid_sp(sp_origin, -lim, lim);
    //     std::cout << "s-type and p-type functions: " << trapezoid_sp(tol) << std::endl;
    // }
    
    return 0;
}