#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <armadillo>
#include <filesystem>
#include "header_file.hpp"

using namespace std;

struct Atom {
    int atomicNumber;
    arma::vec r;
};

ostream& operator << (ostream& os, const Atom& atom) {
    return (os << atom.atomicNumber << "(" << atom.r[0] << ", " << atom.r[1] << ", " << atom.r[2] << ")");
}

struct Simulation {
    arma::vec atomic_numbers;
    arma::mat positions;
    arma::mat forces;
    double sigma;
    double epsilon;
    string kind;

    Simulation(const vector<Atom> atoms, const double SIGMA, const double EPSILON, const string KIND) {
        atomic_numbers = arma::vec(atoms.size());
        positions = arma::mat(3, atoms.size());
        for (int i = 0; i < positions.n_cols; i++) {
            atomic_numbers(i) = atoms[i].atomicNumber;
            positions.col(i) = atoms[i].r;
        }
        sigma = SIGMA;
        epsilon = EPSILON;
        kind = KIND;
    }

    Simulation(const Simulation& sim) {
        atomic_numbers = sim.atomic_numbers;
        positions = sim.positions;
        sigma = sim.sigma;
        epsilon = sim.epsilon;
        kind = sim.kind;
    }

    Simulation(const Simulation& sim, arma::mat forces, double step) {
        atomic_numbers = sim.atomic_numbers;
        positions = sim.positions;
        sigma = sim.sigma;
        epsilon = sim.epsilon;
        kind = sim.kind;
        double total = 0.0;
        for (int i = 0; i < sim.positions.n_cols; i++) {
            for (int j = 0; j < 3; j++) {
                total += pow(forces(j, i), 2);
            }
        }
        step /= sqrt(total);
        for (int i = 0; i < sim.positions.n_cols; i++) {
            for (int j = 0; j < 3; j++) {
                positions(j, i) += step * forces(j, i);
            }
        }
    }

    vector<Atom> atoms() {
        vector<Atom> result;
        for (int i = 0; i < positions.n_cols; i++) {
            Atom atom = { atomic_numbers[i], positions.col(i) };
            result.push_back(atom);
        }
        return result;
    }

    double lennard_jones_energy() {
        double energy = 0.0;
        for (int i = 0; i < positions.n_cols; i++) {
            for (int j = i + 1; j < positions.n_cols; j++) {
                double R_ij = norm(positions.col(i) - positions.col(j));
                energy += epsilon * (pow(sigma / R_ij, 12) - 2 * pow(sigma / R_ij, 6));
            }
        }
        return energy;
    }
    arma::mat lennard_jones_numerical_force(const double h, const bool center) {
        arma::mat forces(positions.n_rows, positions.n_cols);
        double energy = lennard_jones_energy();
        for (int i = 0; i < positions.n_cols; i++) {
            for (int k = 0; k < 3; k++) {
                double energy_left = energy;
                double energy_right = energy;
                for (int j = 0; j < positions.n_cols; j++) {
                    if (i == j) {
                        continue;
                    }
                    double R_ij = norm(positions.col(i) - positions.col(j));
                    double energy_contribution = epsilon * (pow(sigma / R_ij, 12) - 2 * pow(sigma / R_ij, 6));
                    energy_left -= energy_contribution;
                    energy_right -= energy_contribution;
                    arma::vec atom_left = positions.col(i);
                    atom_left[k] -= h;
                    arma::vec atom_right = positions.col(i);
                    atom_right[k] += h;
                    R_ij = norm(atom_left - positions.col(j));
                    energy_left += epsilon * (pow(sigma / R_ij, 12) - 2 * pow(sigma / R_ij, 6));
                    R_ij = norm(atom_right - positions.col(j));
                    energy_right += epsilon * (pow(sigma / R_ij, 12) - 2 * pow(sigma / R_ij, 6));
                }
                if (center) {
                    forces(k, i) = -(energy_right - energy_left) / (2.0 * h);
                }
                else {
                    forces(k, i) = -(energy_right - energy) / h;
                }
            }
        }
        return forces;
    }
    arma::mat lennard_jones_analytic_force() {
        arma::mat forces = arma::zeros(positions.n_rows, positions.n_cols);
        for (int i = 0; i < positions.n_cols; i++) {
            for (int j = 0; j < positions.n_cols; j++) {
                if (i == j) {
                    continue;
                }
                double R_ij = norm(positions.col(i) - positions.col(j));
                double mag = 12 * epsilon * (pow(sigma / R_ij, 12) - pow(sigma / R_ij, 6)) / (R_ij * R_ij);
                forces.col(i) += mag * (positions.col(i) - positions.col(j));
            }
        }
        return forces;
    }
    struct bracket {
        double a;
        double b;
        double c;
    };
    bracket bracket_minimum(double a, double b, const double GOLDEN_RATIO = 1.618304) {
        auto f = [&] (double step) -> double {
            Simulation temp = Simulation(*this, forces, step);
            return temp.lennard_jones_energy();
        };
        double fa = f(a);
        double fb = f(b);
        if (fb > fa) {
            swap(fa, fb);
            swap(a, b);
        }
        double c = b + GOLDEN_RATIO * (b - a);
        double fc = f(c);
        double fu;
        while (fb > fc) {
            double r = (b - a) * (fb - fc);
            double q = (b - c) * (fb - fa);
            double u = b - ((b - c) * q - (b - a) * r) / (2.0 * (q - r));
            double ulim = b + 100.0 * (c - b);
            if ((b - u) * (u - c) > 0.0) {
                fu = f(u);
                if (fu < fc) {
                    a = b;
                    fa = fb;
                    b = u;
                    fb = fu;
                    break;
                }
                else if (fu > fb) {
                    c = u;
                    fc = fu;
                    break;
                }
                u = c + GOLDEN_RATIO * (c - b);
            }
            else if ((c - u) * (u - ulim) > 0.0) {
                if (fu < fc) {
                    fu = f(u);
                    b = c;
                    fb = fc;
                    c = u;
                    fc = fu;
                    u = u + GOLDEN_RATIO * (u - c);
                }
            }
            else {
                u = c + GOLDEN_RATIO * (c - b);
            }
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = u;
            fc = f(u);
        }
        bracket results = {a, b, c};
        return results;
    }
    Simulation line_search(ofstream &fout, double a, double b, const double GOLDEN_RATIO = 1.618304, const double TOL = 3.0e-8, const double h = 1e-4, const double l = 0.3) {
        double xmin = 0.0;
        auto f = [&](double step) -> double {
            Simulation temp = Simulation(*this, forces, step);
            return temp.lennard_jones_energy();
        };

        if (kind == "SD_with_line_search") {
            fout << "Start golden section search" << endl;
        }
        int flag = 0;
        bracket results;
        if (kind == "SD_with_line_search") {
            results = bracket_minimum(a, b, GOLDEN_RATIO);
            flag = 2;
        }
        if (flag == 2) {
            // Do golden section search
            double x0 = 0.0;
            double x1 = results.a;
            double x2 = results.b;
            double x3 = results.c;
            if (abs(x3 - x2) < abs(x1 - x0)) {
                x2 += (x3 - x2) * (1.0 - 1.0 / GOLDEN_RATIO);
            }
            else {
                x1 -= (x1 - x0) * (1.0 - 1.0 / GOLDEN_RATIO);
            }
            double f1 = f(x1);
            double f2 = f(x2);
            while (abs(x3 - x0) > TOL * (abs(x1) + abs(x2))) {
                if (f1 < f2) {
                    x3 = x2;
                    x2 = x1;
                    x1 = x1 / GOLDEN_RATIO + x0 * (1.0 - 1.0 / GOLDEN_RATIO);
                    f2 = f1;
                    f1 = f(x1);
                }
                else {
                    x0 = x1;
                    x1 = x2;
                    x2 = x2 / GOLDEN_RATIO + x3 * (1.0 - 1.0 / GOLDEN_RATIO);
                    f1 = f2;
                    f2 = f(x2);
                }
            }
            if (f1 < f2) {
                xmin = x1;
            }
            else {
                xmin = x2;
            }
        }
        else {
            // Do steepest descent
            xmin = l;
        }
        Simulation sim = Simulation(*this, forces, xmin);
        if (sim.lennard_jones_energy() > this->lennard_jones_energy()) {
            return Simulation(*this);
        }
        else {
            return sim;
        }
    }
};


ostream& operator << (ostream& os, const Simulation& sim) {
    return (os << sim.positions);
}

double calc_error(arma::mat exact, arma::mat value) {
    double total = 0;
    for (int i = 0; i < exact.n_cols; i++) {
        for (int j = 0; j < 3; j++) {
            total += pow(value(j, i) - exact(j, i), 2);
        }
    }
    return sqrt(total);
}
double norm2(arma::mat value) {
    double total = 0;
    for (int i = 0; i < value.n_cols; i++) {
        for (int j = 0; j < value.n_rows; j++) {
            total += pow(value(j, i), 2);
        }
    }
    return sqrt(total);
}

int main() {
    int index = 1;
    const string HW_FOLDER = ".";
    const string INPUT_PATH = HW_FOLDER + "/sample_input";
    for (const auto& kind_result : filesystem::directory_iterator(INPUT_PATH)) {
        const string kind = kind_result.path().stem().string();
        for (const auto& file : filesystem::directory_iterator(kind_result.path())) {
            const string filename = file.path().filename().string();
            const string output_file = HW_FOLDER + "/output/" + kind + "/" + filename;
            ifstream fin(file.path());
            ofstream fout(output_file);
            const double sigma_Au = 2.951; // Angstrom
            const double epsilon_Au = 5.29; // kcal/mol
            const double golden_ratio = 1.618304; // golden ratio
            const double tol = 1e-8;
            vector<Atom> atoms;
            int nAtoms;
            fin >> nAtoms;
            for (int i = 0; i < nAtoms; i++) {
                int atomicNumber;
                double x, y, z;
                fin >> atomicNumber >> x >> y >> z;
                arma::vec r = { x, y, z };
                Atom atom = { atomicNumber, r };
                if (kind == "Energy") {
                    fout << atom << endl;
                }
                if (atomicNumber != 79) {
                    throw invalid_argument("Not gold!");
                }
                atoms.push_back(atom);
            }
            Simulation sim(atoms, sigma_Au, epsilon_Au, kind);
            double energy = sim.lennard_jones_energy();
            if (kind == "Energy" || kind == "Force") {
                fout << "E_LJ = " << energy << endl;
                if (kind == "Force") {
                    fout << "F_LJ_analytical" << endl;
                    arma::mat analytic_forces = sim.lennard_jones_analytic_force();
                    fout << analytic_forces;

                    double forward_error[4];
                    double center_error[4];
                    arma::mat forces;
                    for (int i = 0; i < 4; i++) {
                        double h = pow(10, -i - 1);
                        fout << "Stepsize for finite difference:" << h << endl;
                        fout << "F_LJ_forward_difference" << endl;
                        forces = sim.lennard_jones_numerical_force(h, false);
                        fout << forces;
                        forward_error[i] = calc_error(analytic_forces, forces);
                        fout << "F_LJ_central_difference" << endl;
                        forces = sim.lennard_jones_numerical_force(h, true);
                        fout << forces;
                        center_error[i] = calc_error(analytic_forces, forces);
                    }
                    fout << "Forward MSE:" << forward_error[0] << " " << forward_error[1] << " " << forward_error[2] << " " << forward_error[3] << endl;
                    fout << "Center MSE:" << center_error[0] << " " << center_error[1] << " " << center_error[2] << " " << center_error[3] << endl;
                }
            }
            else if (kind == "SD_with_line_search" || kind == "standard_SD") {
                if (kind == "SD_with_line_search") {
                    fout << "start steepest descent with golden section line search" << endl;
                }
                else {
                    fout << "start steepest descent" << endl;
                }

                fout << "Initial energy: " << energy << endl;

                double h = 0.0001;
                double l = 0.3;
                double force_threshold = 0.01;
                fout << "Stepsize for central difference is:" << h;
                if (kind == "SD_with_line_search") {
                    fout << ";Initial stepsize for line search is:" << l;
                }
                else {
                    fout << ";Initial stepsize for steepest descent is:" << l;
                }
                fout << ";Threshold for convergence in force is:" << force_threshold << endl;

                arma::mat forces;
                if (kind == "standard_SD") {
                    fout << "Analytical Force" << endl;
                    forces = sim.lennard_jones_analytic_force();
                    fout << forces;
                    fout << "Forward Difference Force" << endl;
                    forces = sim.lennard_jones_numerical_force(h, false);
                    fout << forces;
                    fout << "Central Difference Force" << endl;
                    forces = sim.lennard_jones_numerical_force(h, true);
                    fout << forces;
                    fout << "Start steepest descent with central difference force." << endl;
                }
                else {
                    fout << "Central Difference Force" << endl;
                    forces = sim.lennard_jones_numerical_force(h, true);
                    fout << forces;
                    fout << "Start steepest descent with golden section line search using central difference force" << endl;
                }
                sim.forces = forces;
                int iteration = 1;
                for (; iteration < 1000; iteration++) {
                    double old_energy = energy;
                    sim = sim.line_search(fout, 0.0, l, golden_ratio, 3.0e-8, h, l);
                    sim.forces = sim.lennard_jones_numerical_force(h, true);
                    energy = sim.lennard_jones_energy();
                    if (kind == "SD_with_line_search") {
                        fout << "new_point" << endl;;
                        fout << sim;
                        fout << "current energy: " << energy << endl;
                        fout << "Central Difference Force" << endl;
                        fout << sim.forces;
                    }
                    if (norm2(sim.forces) < force_threshold) {
                        break;
                    }
                    if (energy < old_energy) {
                        l *= 1.01;
                    }
                    else {
                        l /= 2.0;
                    }
                }
                fout << "Total iterations: " << iteration << endl;
                energy = sim.lennard_jones_energy();
                fout << "Final energy: " << energy << endl;
                fout << "Optimized structure:" << endl;
                for (Atom atom : sim.atoms()) {
                    fout << atom << endl;
                }
            }
            else {
                throw invalid_argument("Invalid kind of operation!");
            }
        }
    }

    return 0;
}