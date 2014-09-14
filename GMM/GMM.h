
#ifndef __GMM__GMM__
#define __GMM__GMM__

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <fstream>

using namespace std;
using namespace Eigen;
/*!Option structure for the fit function of Gaussian moxiture model.*/ 
struct fitOption {
    string covType = "full";
    /*!<
     *'diagonal': covariance matrices are restricted to be diagonal;

     *'spherical': covariance matrices are restricted to be spherical;

     *'full': covariance matrices are restricted to be full( default).
    */
    bool sharedCov = false;
    /*!<
     *True if the component covariance matrices are restricted to be the same, default is false.
    */
    string start = "kmeans";
    /*!<
     *'random': randomly select from train data to initialize model parameters;

     *'kmeans' use kmeans to get initial clustering index of the train data( default).
    */
    double regularize = 0.0;
    /*!<
     *Regularization parameter to avoid covariance being singular, defalut is 0.
     */
    string display = "iter";
    /*!<
     *Show information during training (default); 'final' show final information.
     */
    unsigned int maxIter = 1000;
    /*!<
     *Maximum iterations allowed, default is 1000;
     */
    double tolFun = 1e-6;
    /*!<
     *The termination tolerance for the log-likelihood function, default is 1e-6.
     */
    bool converged = false;
    /*!<
     *True if fit function converged, false for not converged.
     */
    unsigned int iters = 0;
    /*!<
     *The num of iterations when the fit function converged.
     */
};

/*! 
    * \brief Gaussian mixture model class. Provide fit, cluster and save functionality.
*/
class GMM {
    friend ostream & operator<<(ostream &os, const GMM &gmm);
    friend ifstream & operator>>(ifstream &in, GMM &gmm);
    friend ofstream & operator<<(ofstream &out, const GMM &gmm); 
    long nComponents;/*!< The number of mixture components, K; */
    long nDimensions;/*!< The number of dimensions for each Gaussian component, D; */
    MatrixXd mu;/*!< A K-by-D matrix of component means; */
    vector<MatrixXd> Sigma;/*!< A K-by-D-by-D matrix of component covariance matrices; */
    VectorXd p;/*!< A K-by-1 vector of the proportion of each component. */
    fitOption option;/*!< Containing options about the GMM model. */
    /*! 
        * Check the train data to ensure it is valid.
        * @param data The train data matrix to be checked.
    */
    void checkData(MatrixXd data);
    /*!
        * Check the option to ensure it is valid.
        * @param option The option for fit function to be checked.
    */
    void checkOption(fitOption option);
    /*!
        * Check model to ensure it is valid.
    */
    void checkModel();
    /*!
        * Check if there is Not-a-Number in the matrix
    */
    void checkNaN(MatrixXd mat);
    /*!
        * Initialize model parameters.
        * @param data The train data matrix.
    */
    void initParam(MatrixXd data);
    /*!
        * @param data A N-by-D matrix containing the data;
        * @param lh A N-by-K matrix containing the likelihood,
        * lh(i, j) = Pr(datapoint I | component J)
    */
    void likelihood(MatrixXd data, MatrixXd &lh);
    /*! calculate posterior probability of given data and GMM
        * @param data A N-by-D matrix containing the data;
        * @return The Log-likelihood loss
    */
    double posterior(MatrixXd data, MatrixXd &post);
    
public:
    /// Default constructor
    GMM() = default;
    ///  Constructors with nComponents, nDimensions and option
    GMM(long nComponents, long nDimensions, fitOption option = fitOption());
    ///  Constructors with mu, Sigma, p, and option
    GMM(MatrixXd mu, vector<MatrixXd> Sigma, VectorXd p, fitOption option = fitOption());
    ///  Copy constructors
    GMM(GMM &gmm);
    ///  = operator
    GMM& operator=(const GMM& gmm);
    /*!
        * \brief Train GMM to fit data;
        * @param data A N-by-D matrix containing the train data.
    */
    void fit(MatrixXd data);
    /*!
        * \brief Cluster the with trained model.
        * @param data  A N-by-D matrix containing the test data;
        * @param post  A N-by-K matrix containing the posterior peobability of p(component J | dataPoint I);
        * @return The negative Log likelihood
    */
    double cluster(MatrixXd data, MatrixXd &post);
    /*!
        * \brief Cluster the with trained model.
        * @param data  A N-by-D matrix containing the test data;
        * @param idx  A N-by-1 vector representing N data points' correspoding Gaussian components;
        * @param post  A N-by-K matrix containing the posterior peobability of p(component J | dataPoint I);
        * @return The negative Log likelihood
    */
    double cluster(MatrixXd data, MatrixXd &post, VectorXd &idx);
};

/// Random number generator
int myRandom(int i);
/*! 
    * \brief Random permutation generator.
    * @param n The function permutates 1:n.
*/
VectorXd randperm(unsigned long n);

/*!
    * \brief K-Means class. Provide fit functionality.
*/

void kMeans(MatrixXd data, long nComponents, VectorXd &idx,
        VectorXd &p, MatrixXd &mu, vector<MatrixXd> Sigma);

/*!
    * \brief Read GMM from a file stream.
*/
ifstream & operator>>(ifstream &in, GMM &gmm);
/*!
    * \brief Print GMM.
*/
ostream & operator<<(ostream &os, const GMM &gmm);
/*!
    * \brief write GMM into a file
*/
ofstream & operator<<(ofstream &out, const GMM &gmm);

#endif /* defined(__GMM__GMM__) */
