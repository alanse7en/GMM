//
//  main.cpp
//  GMM
//
//  Created by deng on 14-8-18.
//  Copyright (c) 2014年 deng. All rights reserved.
//

#include <iostream>
#include "GMM.h"
#include <vector>
#include <ctime>
#include <math.h>


int main(int argc, const char * argv[])
{

    fitOption option;
    option.covType = "spherical";
    option.start = "random";
    GMM gmmTest(2, 10, option);
    MatrixXd data = MatrixXd::Random(400, 10);
    gmmTest.fit(data);
}

