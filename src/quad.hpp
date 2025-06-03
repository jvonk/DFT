#pragma once

#include <cmath>

class Quadrature
{
  public:
    int refinement;
    virtual double refine() = 0;
    double operator()(const double tol, const int MAXSTEPS = 40)
    {
        double oldvalue;
        for (int i = 0; i < MAXSTEPS; i++) {
            double value = refine();
            if ((i > 0) && ((fabs(value - oldvalue) < tol) ||
                            (value == 0 && oldvalue == 0))) {
                return value;
            }
            oldvalue = value;
        }
        throw("Quadrature did not converge, too many steps.");
    }
};

template <class T> class Trapezoid : Quadrature
{
  private:
    double lower, upper, value;

  public:
    T &f;
    Trapezoid() {}
    Trapezoid(T &F, const double LOWER, const double UPPER)
        : f(F), lower(LOWER), upper(UPPER)
    {
        refinement = 0;
    }
    double refine()
    {
        if (refinement++ == 0) {
            value = (upper - lower) * (f(lower) + f(upper)) / 2.0;
        } else {
            int nsteps = 1 << (refinement - 2);
            double del = (upper - lower) / nsteps;
            double x = lower + del / 2.0;
            double sum = 0.0;
            for (int i = 0; i < nsteps; i++, x += del) {
                sum += f(x);
            }
            value = (value + (upper - lower) * sum / nsteps) / 2.0;
        }
        return value;
    }
    double operator()(const double tol, const int MAXSTEPS = 20)
    {
        return Quadrature::operator()(tol, MAXSTEPS);
    }
};

class Function
{
  public:
    virtual double operator()(double x) = 0;
};

class DoubleGaussian : Function
{
  private:
    double XA, XB, alpha, beta;
    int lA, lB;

  public:
    DoubleGaussian(const double XA_in, const int lA, const double alpha,
                   const double XB, const int lB, const double beta)
        : XA(XA), lA(lA), alpha(alpha), XB(XB), lB(lB), beta(beta)
    {
    }
    double operator()(double x)
    {
        double dxA = x - XA;
        double dxB = x - XB;
        return pow(dxA, lA) * pow(dxB, lB) *
               exp(-alpha * dxA * dxA - beta * dxB * dxB);
    }
};