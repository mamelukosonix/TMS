#include "TMO.h"

class TMOZhao18 : public TMO
{
public:
    TMOZhao18();
    virtual ~TMOZhao18();
    virtual int Transform();

protected:
    TMODouble sigma;  
    TMODouble mu;
    TMODouble lambda0;
    TMODouble downsample;
};
