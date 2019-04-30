#ifndef QUANTIZATION_H
#define QUANTIZATION_H

class Quantization
{
public:
    virtual void init() = 0;
    virtual void iterate() = 0;
    virtual void finalize() = 0;
};

#endif // QUANTIZATION_H