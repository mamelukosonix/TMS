/************************************************************************************
*                                                                                   *
*                       Brno University of Technology                               *
*                       CPhoto@FIT                                                  *
*                                                                                   *
*                       Tone Mapping Studio                                         *
*                                                                                   *
*                                                                                   *
*                       Author: Robert Zelníček [xzelni06@stud.fit.vutbr.cz]        *
*                       Brno 2025                                                   *
*                                                                                   *
*                       Implementation of the TMOLiu19 class                    *
*                                                                                   *
************************************************************************************/

#include "TMOLiu19.h"
#include "TMOImage.h"

#include <vector>

// --------------------------------------------------------------------------- //
// TMOLiu19: ctor/params
// --------------------------------------------------------------------------- //
TMOLiu19::TMOLiu19()
{
    SetName(L"Liu19");
    SetDescription(L"Color-to-gray conversion with perceptual preservation "
                   L"and dark channel prior (Liu et al., 2019).");

    // --- Model parameters ---
    alpha1.SetName(L"alpha1");
    alpha1.SetDescription(L"Weight of brightness fidelity term");
    alpha1.SetDefault(1.0);
    alpha1 = 0.0;
    alpha1.SetRange(0.0, 10.0);
    this->Register(alpha1);

    alpha2.SetName(L"alpha2");
    alpha2.SetDescription(L"Weight of contrast consistency term");
    alpha2.SetDefault(1.0);
    alpha2 = 0.0;
    alpha2.SetRange(0.0, 10.0);
    this->Register(alpha2);

    eta.SetName(L"eta");
    eta.SetDescription(L"Weight of dark channel prior");
    eta.SetDefault(0.1);
    eta = 0.0;
    eta.SetRange(0.0, 10.0);
    this->Register(eta);

    beta0.SetName(L"beta0");
    beta0.SetDescription(L"Initial penalty parameter beta");
    beta0.SetDefault(0.01);
    beta0 = 0.0;
    beta0.SetRange(0.0, 10.0);
    this->Register(beta0);

    gamma0.SetName(L"gamma0");
    gamma0.SetDescription(L"Initial penalty parameter gamma");
    gamma0.SetDefault(0.01);
    gamma0 = 0.0;
    gamma0.SetRange(0.0, 10.0);
    this->Register(gamma0);

    maxIter.SetName(L"Iterations");
    maxIter.SetDescription(L"Maximum number of iterations");
    maxIter.SetDefault(50);
    maxIter = 0;
    maxIter.SetRange(1, 1000);
    this->Register(maxIter);
}

TMOLiu19::~TMOLiu19() = default;

// --------------------------------------------------------------------------- //
// Transform
// --------------------------------------------------------------------------- //
int TMOLiu19::Transform()
{
    pSrc->Convert(TMO_RGB);
    pDst->Convert(TMO_RGB);

    const int width  = pSrc->GetWidth();
    const int height = pSrc->GetHeight();

    double *src = pSrc->GetData();
    double *dst = pDst->GetData();

    // STEP 1: Convert input image to perceptual components (p1, p2, p3)
    // TODO:
    //  - Convert RGB to CIE Lab
    //  - Extract brightness p1 and color components p2, p3
    //  - Store them in vectors of size N

    // STEP 2: Initialize variables g, f, w (Algorithm 2, initialization)
    // TODO:
    //  - Initialize g^0 (grayscale image)
    //  - Initialize f^0 and w^0
    //  - Initialize beta = beta0, gamma = gamma0

    // STEP 3: Alternating minimization loop
    for (int iter = 0; iter < maxIter; ++iter)
    {
        pSrc->ProgressBar(iter, maxIter);

        // TODO:
        //  - Compute dark channel operator M
        //  - Solve normal equations for g^{k+1}
        //  - beta *= 2
        //  - gamma *= 2
    }
    double *dst = pDst->GetData();

    // TODO:
    //  - Copy final grayscale into rgb channels

    pSrc->ProgressBar(maxIter, maxIter);
    return 0;
}
