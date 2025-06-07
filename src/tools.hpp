#pragma once

#include <armadillo>
#include <cmath>
#include <iomanip>
#include <iostream>

double calc_error(const arma::mat &exact, const arma::mat &value)
{
    double total = 0;
    for (arma::uword i = 0; i < exact.n_cols; i++) {
        for (int j = 0; j < 3; j++) {
            total += pow(value(j, i) - exact(j, i), 2);
        }
    }
    return sqrt(total);
}

double norm2(const arma::mat &value)
{
    double total = 0;
    for (arma::uword i = 0; i < value.n_cols; i++) {
        for (int j = 0; j < value.n_rows; j++) {
            total += pow(value(j, i), 2);
        }
    }
    return sqrt(total);
}

int factorial(int n)
{
    int product = 1;
    for (int i = 1; i <= n; i++) {
        product *= i;
    }
    return product;
}
int double_factorial(int n)
{
    int product = 1;
    for (int i = n; i > 0; i -= 2) {
        product *= i;
    }
    return product;
}
double binomial(int m, int n)
{
    return factorial(m) / (factorial(n) * factorial(m - n));
}


int n_functions(const int l) { return (l + 1) * (l + 2) / 2; }

arma::umat lmat(const int l)
{
    arma::uword count = n_functions(l);
    arma::umat result(3, count);
    for (arma::uword i = 0; i <= l; i++) {
        for (arma::uword j = 0; i + j <= l; j++) {
            count--;
            result(0, count) = i;
            result(1, count) = j;
            result(2, count) = l - i - j;
        }
    }
    return result;
}

arma::mat density_matrix(const arma::mat &C)
{
    arma::mat P = arma::zeros(C.n_rows, C.n_rows);
    for (arma::uword i = 0; i < C.n_cols; i++) {
        P += C.col(i) * C.col(i).t();
    }
    return P;
}