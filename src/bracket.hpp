#pragma once

#include <iomanip>
#include <iostream>

struct bracket {
    double a;
    double b;
    double c;
};

bracket bracket_minimum(std::function<double(double)> f, double a, double b,
                        const double GOLDEN_RATIO = 1.618304)
{
    double fa = f(a);
    double fb = f(b);
    if (fb > fa) {
        std::swap(fa, fb);
        std::swap(a, b);
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
            } else if (fu > fb) {
                c = u;
                fc = fu;
                break;
            }
            u = c + GOLDEN_RATIO * (c - b);
        } else if ((c - u) * (u - ulim) > 0.0) {
            if (fu < fc) {
                fu = f(u);
                b = c;
                fb = fc;
                c = u;
                fc = fu;
                u = u + GOLDEN_RATIO * (u - c);
            }
        } else {
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

double line_search(std::ostream &fout, std::function<double(double)> f,
                   double a, double b, const double GOLDEN_RATIO = 1.618304,
                   const double TOL = 3.0e-8, const double h = 1e-4,
                   const double l = 0.3, const bool golden_section = true)
{
    double xmin = 0.0;

    if (golden_section) {
        fout << "Start golden section search" << std::endl;
    }
    int flag = 0;
    bracket results;
    if (golden_section) {
        results = bracket_minimum(f, a, b, GOLDEN_RATIO);
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
        } else {
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
            } else {
                x0 = x1;
                x1 = x2;
                x2 = x2 / GOLDEN_RATIO + x3 * (1.0 - 1.0 / GOLDEN_RATIO);
                f1 = f2;
                f2 = f(x2);
            }
        }
        if (f1 < f2) {
            xmin = x1;
        } else {
            xmin = x2;
        }
    } else if (f(l) <= f(xmin)) {
        xmin = l;
    }
    return xmin;
}