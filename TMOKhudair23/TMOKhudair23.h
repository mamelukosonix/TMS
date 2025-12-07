#ifndef _TMOKHUDAIR23_H_
#define _TMOKHUDAIR23_H_

#include "TMO.h"

class TMOKhudair23 : public TMO
{
public:
    TMOKhudair23();
    virtual ~TMOKhudair23();

    // main operator implementation
    virtual int Transform();

protected:
    // This will be the 'k' from the paper: G = ||S|| / k
    TMODouble dParameter;
};

#endif
