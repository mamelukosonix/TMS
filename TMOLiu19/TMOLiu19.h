#ifndef _TMOLIU19_H_
#define _TMOLIU19_H_

#include "TMO.h"

class TMOLiu19 : public TMO
{
public:
    TMOLiu19();
    virtual ~TMOLiu19();
    virtual int Transform();

protected:
    TMODouble alpha1;    // contrast weight for p1 (L)
    TMODouble alpha2;    // contrast weight for p2 and p3 (a,b), alpha2 = alpha3
    TMODouble eta;       // dark channel prior weight

    TMODouble beta0;     // initial beta (0 -> uses 2*eta as in paper)
    TMODouble gamma0;    // initial gamma (0 -> uses 2*eta as in paper)

    // --- iteration / solver controls ---
    TMOInt Iterations;   // safety cap on iterations
    TMOInt PatchRadius;  // dark-channel patch radius N(x)
    TMOInt CGIters;      // CG iterations for solving Eq.(10)
    TMODouble CGTol;     // CG tolerance
};

#endif
