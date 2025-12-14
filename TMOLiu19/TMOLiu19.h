#include "TMO.h"
class TMOLiu19 : public TMO
{
public:
    TMOLiu19();
    virtual ~TMOLiu19();

    virtual int Transform();

protected:
    TMODouble alpha1;   // weight for brightness fidelity
    TMODouble alpha2;   // weight for contrast term
    TMODouble eta;      // dark channel prior weight

    TMODouble beta0;    // initial penalty parameter
    TMODouble gamma0;   // initial penalty parameter

    TMOInt    maxIter;  // maximum number of iterations
};
