//
//  main.cpp
//  GMM
//
//  Created by deng on 14-8-18.
//  Copyright (c) 2014年 deng. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "GMM.h"
#include <vector>
#include <ctime>
#include <math.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

/*
 We need a functor that can pretend it's const,
 but to be a good random number generator
 it needs mutable state.
 */
namespace Eigen {
namespace internal {
template<typename Scalar> struct scalar_normal_dist_op
        {
            static boost::mt19937 rng;    // The uniform pseudo-random algorithm
            mutable boost::normal_distribution<Scalar> norm;  // The gaussian combinator
            
            EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)
            
            template<typename Index>
            inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
        };
        
        template<typename Scalar> boost::mt19937 scalar_normal_dist_op<Scalar>::rng;
        
        template<typename Scalar>
        struct functor_traits<scalar_normal_dist_op<Scalar> >
        { enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };
    } // end namespace internal
} // end namespace Eigen

MatrixXd generateData(){
    int size_1 = 2; // Dimensionality (rows)
    int nn_1= 499;     // How many samples (columns) to draw
    Eigen::internal::scalar_normal_dist_op<double> randN_1; // Gaussian functor
    Eigen::internal::scalar_normal_dist_op<double>::rng.seed(1); // Seed the rng
    
    // Define mean and covariance of the distribution
    Eigen::VectorXd mean_1(size_1);
    Eigen::MatrixXd covar_1(size_1,size_1);
    
    mean_1  <<  0,  0;
    covar_1 <<  1.5, 0,
    0,  1.5;
    
    Eigen::MatrixXd normTransform_1(size_1,size_1);
    
    Eigen::LLT<Eigen::MatrixXd> cholSolver_1(covar_1);
    
    // We can only use the cholesky decomposition if
    // the covariance matrix is symmetric, pos-definite.
    // But a covariance matrix might be pos-semi-definite.
    // In that case, we'll go to an EigenSolver
    if (cholSolver_1.info()==Eigen::Success) {
        // Use cholesky solver
        normTransform_1 = cholSolver_1.matrixL();
    } else {
        // Use eigen solver
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver_1(covar_1);
        normTransform_1 = eigenSolver_1.eigenvectors()
        * eigenSolver_1.eigenvalues().cwiseSqrt().asDiagonal();
    }
    
    Eigen::MatrixXd samples_1 = (normTransform_1
                                 * Eigen::MatrixXd::NullaryExpr(size_1,nn_1,randN_1)).colwise()
    + mean_1;
    
    int size_2 = 2; // Dimensionality (rows)
    int nn_2= 500;     // How many samples (columns) to draw
    Eigen::internal::scalar_normal_dist_op<double> randN_2; // Gaussian functor
    Eigen::internal::scalar_normal_dist_op<double>::rng.seed(1); // Seed the rng
    
    // Define mean and covariance of the distribution
    Eigen::VectorXd mean_2(size_2);
    Eigen::MatrixXd covar_2(size_2,size_2);
    
    mean_2  <<  7,  3;
    covar_2 <<  1.5, 0,
    0,  1.5;
    
    Eigen::MatrixXd normTransform_2(size_2,size_2);
    
    Eigen::LLT<Eigen::MatrixXd> cholSolver_2(covar_2);
    
    // We can only use the cholesky decomposition if
    // the covariance matrix is symmetric, pos-definite.
    // But a covariance matrix might be pos-semi-definite.
    // In that case, we'll go to an EigenSolver
    if (cholSolver_2.info()==Eigen::Success) {
        // Use cholesky solver
        normTransform_2 = cholSolver_2.matrixL();
    } else {
        // Use eigen solver
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver_2(covar_2);
        normTransform_2 = eigenSolver_2.eigenvectors()
        * eigenSolver_2.eigenvalues().cwiseSqrt().asDiagonal();
    }
    
    Eigen::MatrixXd samples_2 = (normTransform_2
                                 * Eigen::MatrixXd::NullaryExpr(size_2,nn_2,randN_2)).colwise()
    + mean_2;
    
    Eigen::MatrixXd data(samples_1.cols()+samples_2.cols(), 2);
    data.block(0, 0, samples_1.cols(), 2) = samples_1.transpose();
    data.block(samples_1.cols(), 0, samples_2.cols(), 2) = samples_2.transpose();
    return data;
}
int main(int argc, const char * argv[])
{
    MatrixXd data = generateData();
    fitOption option;
    option.covType = "diagonal";
    option.start = "random";
    option.display = "iter";
    option.maxIter = 1e4;
    option.tolFun = 1e-10;
    option.sharedCov = true;
//    option.regularize = 1e-4;
    int nComponents = 2;
    int nDimensions = 2;
    ShaSpheGMM gmmTest(nComponents, nDimensions, option);
//    GMM gmm("/Users/Apple/Dropbox/xuesong.deng/research/GMM/result.xml");
//    GMM gmm;
//    ifstream in("/Users/Apple/Dropbox/xuesong.deng/research/GMM/result.xml");
//    in >> gmm;
////    MatrixXd data2 = MatrixXd::Random(3, 1);
    gmmTest.fit(data);
    ShaSpheGMM gmmTest2 = gmmTest;
    ofstream out("/Users/Apple/Dropbox/xuesong.deng/research/GMM/result.xml");
    out << gmmTest2;
    MatrixXd post = MatrixXd::Zero(data.rows(), nComponents);
    gmmTest.cluster(data, post);
    cout << gmmTest << endl;
//    MatrixXd post = MatrixXd::Zero(data.rows(), 2);
//    VectorXd idx = VectorXd::Zero(data.rows());
//    auto nLogL = gmmTest.cluster(data, post, idx);
//    printf("The negative Log-likelihood is %f \n", nLogL);
//    gmmTest.save("/Users/Apple/Dropbox/xuesong.deng/research/GMM/result.xml");
}

