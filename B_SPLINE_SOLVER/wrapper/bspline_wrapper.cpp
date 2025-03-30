#include "BSplineGeometry.h"
// bindings.cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h> 
#include <iostream>
#include <ceres/ceres.h>
#include <vector>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm> // pour reverse
using namespace std;

namespace nb = nanobind;
using namespace nb;
using namespace nanobind::literals;
template <typename T>
double BSplineGeometry<T>::basisFunction(int p, int i, double t)
{
    if (p == 0)
    {
        if (knot_vector_[i] <= t && t < knot_vector_[i + 1])
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        double bf1 = 0;
        double bf2 = 0;
        if (knot_vector_[i + p] != knot_vector_[i])
        {
            bf1 = (t - knot_vector_[i]) / (knot_vector_[i + p] - knot_vector_[i]) * basisFunction(p - 1, i, t);
        }

        if (knot_vector_[i + p + 1] != knot_vector_[i + 1])
        {
            bf2 = (knot_vector_[i + p + 1] - t) / (knot_vector_[i + p + 1] - knot_vector_[i + 1]) * basisFunction(p - 1, i + 1, t);
        }

        return bf1 + bf2;
    }
}

template <typename T>
void BSplineGeometry<T>::generateSpline(int p)
{
    vector<pair<T, T>> resulting_vector;
    int newNPoints = 0;
    ofstream file("BSpline_curve.txt");

    for (int j = 0; j < npoints_; j++)
    {
        double t = static_cast<double>(j) / (npoints_);
        T x = static_cast<T>(0);
        T y = static_cast<T>(0);

        for (int i = 0; i < control_points_.size(); i++)
        {
            if (j == npoints_ - 1 or j == 0)
            {
                x = static_cast<T>(1);
                y = static_cast<T>(0);
            }
            else
            {
                double basis = basisFunction(p, (i), t);
                x += basis * control_points_[i].first;
                y += basis * control_points_[i].second;
            }
        }
        if (x <= 1)
        {
            file << x << " " << y << endl;
            resulting_vector.push_back({ x,y });
            newNPoints++;
        }
    }
    file.close();
    curve_ = resulting_vector;
    npoints_ = newNPoints;
}

template <typename T>
double BSplineGeometry<T>::getSurface(double x_target, bool upper)
{
    double y_target = 0;
    if (not upper)
    {
        double x_0 = 0, y_0 = 0, x_1 = 0, y_1 = 0;
        for (int i = 0; i < curve_.size(); i++)
        {
            if (curve_[i].first == x_target)
            {
                y_target = curve_[i].second;
                break;
            }
            else if (curve_[i].first > x_target)
            {
                x_1 = curve_[i].first;
                y_1 = curve_[i].second;
            }
            else if (curve_[i].first < x_target)
            {
                x_0 = curve_[i].first;
                y_0 = curve_[i].second;
                y_target = y_0 + (x_target - x_0) * (y_1 - y_0) / (x_1 - x_0);
                break;
            }
        }
    }


    if (upper)
    {
        double x_0 = 0, y_0 = 0, x_1 = 0, y_1 = 0;
        int j = 0;
        for (int i = 0; i < curve_.size(); i++)
        {
            if (curve_[i].first == x_target and j == 1)
            {
                y_target = curve_[i].second;
                break;
            }
            else if (curve_[i].first < x_target)
            {
                x_0 = curve_[i].first;
                y_0 = curve_[i].second;
                j = 1;
            }
            else if (curve_[i].first > x_target and j == 1)
            {
                x_1 = curve_[i].first;
                y_1 = curve_[i].second;
                y_target = y_0 + (x_target - x_0) * (y_1 - y_0) / (x_1 - x_0);
                break;
            }
        }
    }

    return y_target;
}

template <typename T>
pair<vector<double>, vector<double>> BSplineGeometry<T>::getTotalSurface(vector<double> xTarget)
{
    vector<double> yTargetUpper;
    vector<double> yTargetLower;
    vector<double> xFinalUpper;
    vector<double> xFinalLower;

    for (double xc : xTarget)
    {
        if (xc == 1)
        {
            yTargetLower.emplace_back(0);
            xFinalLower.emplace_back(xc);
        }
        else if (xc == 0)
        {
            yTargetUpper.emplace_back(0);
            xFinalUpper.emplace_back(xc);
        }
        else
        {
            yTargetLower.emplace_back(getSurface(xc, false));
            yTargetUpper.emplace_back(getSurface(xc, true));
            xFinalLower.emplace_back(xc);
            xFinalUpper.emplace_back(xc);
        }
    }

    reverse(yTargetUpper.begin(), yTargetUpper.end());
    reverse(xFinalUpper.begin(), xFinalUpper.end());
    yTargetLower.insert(yTargetLower.end(), yTargetUpper.begin(), yTargetUpper.end());
    xFinalLower.insert(xFinalLower.end(), xFinalUpper.begin(), xFinalUpper.end());
    vector<double> yTarget = yTargetLower;
    vector<double> xFinal = xFinalLower;

    return { xFinal, yTarget };
}

vector <double> createKnotVector(int nKnotsDesired, int p)
{
    vector<double> resultingVector;
    double step = 1.0 / (nKnotsDesired - (p + 1) * 2 + 1);
    for (int i = 0; i < nKnotsDesired; i++)
    {
        if (i < (p + 1))
        {
            resultingVector.push_back(0);
        }
        else if (i >= (p + 1) && i < nKnotsDesired - (p + 1))
        {
            resultingVector.push_back(step * (i - (p + 1) + 1));
        }
        else
        {
            resultingVector.push_back(1);
        }
    }
    return resultingVector;
}

//Using Ceres Solver
struct BSplineResidual
{
    BSplineResidual(vector<double> reducedKnotVector, vector<pair<double, double> > initialCurve, double npoints, int p, int i, int nControlPoints)
        : reducedKnotVector_(reducedKnotVector), curve_(initialCurve), npoints_(npoints), p_(p), i_(i), nControlPoints_(nControlPoints) {
    }


    template <typename T>
    bool operator()(const T* const reducedControlPoints, T* residual) const {
        // Transformation de reducedControlPoints en un vector <T, T>
        vector<pair<T, T>> reducedControlPointsVector;
        for (int i = 0; i < nControlPoints_; i++)
        {
            reducedControlPointsVector.emplace_back(reducedControlPoints[2 * i], reducedControlPoints[2 * i + 1]);
        }

        BSplineGeometry<T> finalSpline(reducedKnotVector_, npoints_, reducedControlPointsVector);
        finalSpline.generateSpline((p_));

        if (finalSpline.curve_.size() != curve_.size())
        {
            cout << "The sizes of the initial and final curve don't match. " << endl;
            return false;
        }

        residual[0] = static_cast<T>(curve_[i_].first) - finalSpline.curve_[i_].first;
        residual[1] = static_cast<T>(curve_[i_].second) - finalSpline.curve_[i_].second;

        return true;
    }

    vector<double> reducedKnotVector_;
    vector<pair<double, double>> curve_;
    double npoints_;
    int p_;
    int i_;
    int nControlPoints_;
};

vector <double> createEmptyControlPoints(int nControlPoints)
{
    vector<double> controlPointsVector;
    for (int i = 0; i < nControlPoints; i++)
    {
        controlPointsVector.emplace_back(0);
        controlPointsVector.emplace_back(0);
    }
    return controlPointsVector;
}

vector <pair<double, double>> readFileOihan(string filePath)
{
    string line;
    vector <pair<double, double>> curveOihanUpper;
    vector <pair<double, double>> curveOihanLower;

    ifstream fileOihan(filePath);
    if (!fileOihan)
    {
        cout << "Error in the opening of the file." << endl;
    }
    bool header = false;
    while (getline(fileOihan, line))
    {
        if (line.find("XU") != string::npos)
        {
            header = true;
            continue;
        }

        if (header)
        {
            istringstream ss(line);
            double xU, yU, xL, yL;
            ss >> xU >> yU >> xL >> yL;

            if (ss)
            {
                curveOihanUpper.emplace_back(xU, yU);
                curveOihanLower.emplace_back(xL, yL);
            }

            else
            {
                cout << "Error extracting from line." << endl;
                break;
            }
        }
    }
    reverse(curveOihanLower.begin(), curveOihanLower.end());
    curveOihanLower.insert(curveOihanLower.end(), curveOihanUpper.begin(), curveOihanUpper.end());
    vector <pair<double, double>> curveOihanFull = curveOihanLower;

    return curveOihanFull;
}

vector <pair<double, double>> readFileCurve(string filePath)
{
    vector <pair<double, double>> curve;
    string line;

    ifstream file(filePath);
    if (!file)
    {
        cout << "Error in the opening of the file." << endl;
    }

    while (getline(file, line))
    {
        istringstream ss(line);
        double x, y;
        ss >> x >> y;

        if (ss)
        {
            curve.emplace_back(x, y);
        }

        else
        {
            cout << "Error extracting from line." << endl;
            break;
        }

    }

    return curve;
}

template <typename T>
BSplineGeometry<T>::BSplineGeometry(vector<double> knot_vector, double npoints, vector<std::pair<T, T> > control_points)
{
    knot_vector_ = knot_vector;
    npoints_ = npoints;
    control_points_ = control_points;
}

void run_bspline_solver(std::string filePath, int nKnotsDesired) 
{
    // INPUT
    vector<pair<double, double>> importedCurve = readFileCurve(filePath);

    //CURVE FITTING
    const int p = 3;
    vector<double> newKnotVector = createKnotVector(nKnotsDesired, p);
    const int nControlPoints = nKnotsDesired - p - 1;
    vector<double> newControlPoints = createEmptyControlPoints(nControlPoints);

    //Ceres optimisation with options for nKnotsDesired
    if (nKnotsDesired == 15)
    {
        ceres::Problem problem;
        for (int i = 0; i < importedCurve.size(); i++)
        {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<BSplineResidual, 2, (15 - p - 1) * 2>
                (new BSplineResidual(newKnotVector, importedCurve, importedCurve.size(), p, i, nControlPoints));

            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1), newControlPoints.data());
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        options.max_num_iterations = 100;  // Increase iterations
        options.function_tolerance = 1e-12;  // Reduce convergence tolerance
        options.parameter_tolerance = 1e-10;
        options.trust_region_strategy_type = ceres::DOGLEG;

        ceres::Solver::Summary resume;
        ceres::Solve(options, &problem, &resume);

        cout << resume.FullReport() << endl;
    }
    if (nKnotsDesired == 20)
    {
        ceres::Problem problem;
        for (int i = 0; i < importedCurve.size(); i++)
        {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<BSplineResidual, 2, (20 - p - 1) * 2>
                (new BSplineResidual(newKnotVector, importedCurve, importedCurve.size(), p, i, nControlPoints));

            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1), newControlPoints.data());
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        options.max_num_iterations = 100;  // Increase iterations
        options.function_tolerance = 1e-12;  // Reduce convergence tolerance
        options.parameter_tolerance = 1e-10;
        options.trust_region_strategy_type = ceres::DOGLEG;

        ceres::Solver::Summary resume;
        ceres::Solve(options, &problem, &resume);

        cout << resume.FullReport() << endl;
    }
    if (nKnotsDesired == 25)
    {
        ceres::Problem problem;
        for (int i = 0; i < importedCurve.size(); i++)
        {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<BSplineResidual, 2, (25 - p - 1) * 2>
                (new BSplineResidual(newKnotVector, importedCurve, importedCurve.size(), p, i, nControlPoints));

            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1), newControlPoints.data());
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        options.max_num_iterations = 100;  // Increase iterations
        options.function_tolerance = 1e-12;  // Reduce convergence tolerance
        options.parameter_tolerance = 1e-10;
        options.trust_region_strategy_type = ceres::DOGLEG;

        ceres::Solver::Summary resume;
        ceres::Solve(options, &problem, &resume);

        cout << resume.FullReport() << endl;
    }
    if (nKnotsDesired == 30)
    {
        ceres::Problem problem;
        for (int i = 0; i < importedCurve.size(); i++)
        {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<BSplineResidual, 2, (30 - p - 1) * 2>
                (new BSplineResidual(newKnotVector, importedCurve, importedCurve.size(), p, i, nControlPoints));

            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1), newControlPoints.data());
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        options.max_num_iterations = 100;  // Increase iterations
        options.function_tolerance = 1e-12;  // Reduce convergence tolerance
        options.parameter_tolerance = 1e-10;
        options.trust_region_strategy_type = ceres::DOGLEG;

        ceres::Solver::Summary resume;
        ceres::Solve(options, &problem, &resume);

        cout << resume.FullReport() << endl;
    }

    // Changing format of newControlPoints
    vector <pair<double, double>> newControlPointsVector;
    for (int i = 0; i < nControlPoints; i++)
    {
        newControlPointsVector.emplace_back(newControlPoints[2 * i], newControlPoints[2 * i + 1]);
    }

    BSplineGeometry<double> splineFitted(newKnotVector, 1100, newControlPointsVector);
    splineFitted.generateSpline(p);

    //GETTOTALSURFACE TESTING

    // vector<double> xTarget = {1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64, 0.62,
    //     0.60, 0.58, 0.56, 0.54, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22,
    //         0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.0};
    // auto [xFinal, yTarget] = splineFitted.getTotalSurface(xTarget);
    // cout << "Courbe getSurface" << endl;
    // for (int i = 0; i < xFinal.size(); i++)
    // {
    //     cout << "[" << xFinal[i] << " , " << yTarget[i] << "]," << endl;
    // }

    
}


NB_MODULE(BSpline_solver, m) {
    m.def("run_bspline_solver", &run_bspline_solver, "file_path"_a, "n_knots_desired"_a);
}