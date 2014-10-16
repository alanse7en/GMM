//
//  main.cpp
//  GMM
//
//  Created by deng on 14-8-18.
//  Copyright (c) 2014年 deng. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <math.h>
#include "GMM.h"
#include "GenerateData.h"

int main(int argc, const char * argv[])
{
    MatrixXd data = generateData();
    fitOption option;
    option.start = "kmeans";
    option.display = "iter";
    option.maxIter = 1e4;
    option.tolFun = 1e-10;
//    option.regularize = 1e-4;
    int nComponents = 2;
    int nDimensions = 2;
    ShaFullGMM gmmTest(nComponents, nDimensions, option);
    gmmTest.fit(data);
    cout << gmmTest << endl;
}

