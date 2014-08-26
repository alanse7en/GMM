//
//  GMM.h
//  GMM
//
//  Created by deng on 14-8-18.
//  Copyright (c) 2014年 deng. All rights reserved.
//

#ifndef __GMM__GMM__
#define __GMM__GMM__

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace Eigen;

/*!
     *covType : 'diagonal' covariance matrices are restricted to be diagonal;
     *          'spherical' covariance matrices are restricted to be spherical;
     *          'full' covariance matrices are restricted to be full( default).
     *sharedCov : True if the component covariance matrices are restricted to be the same, default is false.
     *start : 'random' randomly select from train data to initialize model parameters;
     *        'kmeans' use kmeans to get initial clustering index of the train data( default).
     *regularize : Regularization parameter to avoid covariance being singular, defalut is 0.
     *display : 'iter' show information during training (default); 'final' show final information.
     *maxIter : Maximum iterations allowed, default is 100;
     *tolFun : The termination tolerance for the log-likelihood function, default is 1e-6.
     *converged : 'true' GMM training process converged; otherwise failed.
     *iters : The num of iterations to finished training.
*/
struct fitOption {
    string covType = "full";
    bool sharedCov = false;
    string start = "kmeans";
    double regularize = 0.0;
    string display = "iter";
    unsigned int maxIter = 100;
    double tolFun = 1e-6;
    bool converged = false;
    unsigned int iters = 0;
};

    /*!
     *nComponents The number of mixture components, K;
     *nDimensions The number of dimensions for each Gaussian component, D;
     *mu A K-by-D matrix of component means;
     *Sigma A K-by-D-by-D matrix of component covariance matrices;
     *p A K-by-1 vector of the proportion of each component.
     *option Containing options about the GMM model.
     */
class GMM {
    long nComponents;
    long nDimensions;
    MatrixXd mu;
    vector<MatrixXd> Sigma;
    VectorXd p;
    fitOption option;
    /// Check the train data to ensure it is valid.
    void checkData(MatrixXd data);
    /// Check the option to ensure it is valid.
    void checkOption(fitOption option);
    /// Check model to ensure it is valid.
    void checkModel();
    /// Initialize model parameters.
    void initParam(MatrixXd data);
    
public:
    ///  Constructors
    GMM(long nComponents, long nDimensions, fitOption option = fitOption());
    GMM(MatrixXd mu, vector<MatrixXd> Sigma, VectorXd p, fitOption option = fitOption());
    ///  Load GMM from fileName
    GMM(string fileName);
    ///  Copy constructors
    GMM(GMM &gmm);
    ///  = operator
    GMM& operator=(const GMM& gmm);
    /*!
     *Train GMM to fit data with option opt;
     *data A N-by-D matrix containing the train data.
     */
    void fit(MatrixXd data);
    /*!
     *data A N-by-D matrix containing the test data;
     *idx A N-by-1 vector representing N data points' correspoding Gaussian components;
     *nLogL The negative of the log-likelihood of the data;
     *post A N-by-K matrix containing the posterior peobability of p(component J | dataPoint I);
     *logPdf A N-by-1 vector containing estimates of the PDF of dataPoint I, the PDF of I is the
     *sum of p(dataPoint I | component J) * Pr(component J).
     */
    void cluster(MatrixXd data, VectorXd &idx, int &nLogL, MatrixXd &post, VectorXd &logPdf);
    ///  Save GMM to fileName
    void save(string fileName);
};

/// Generate random permutation
int myRandom(int i);
VectorXd randperm(unsigned long n);
#endif /* defined(__GMM__GMM__) */
