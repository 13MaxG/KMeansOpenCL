#ifndef CPU_QUANTIZATION_H
#define CPU_QUANTIZATION_H

#include <vector>
#include <limits>

#include "image.h"

class KMeansCPUQuantization : public Quantization
{
private:
    Image* inputImage;
    Image* outputImage;
    unsigned int colorTableSize;

    std::vector<unsigned char> colorTable;
    std::vector<unsigned int> acc;
    std::vector<unsigned int> counter;
public:
    KMeansCPUQuantization(Image *inputImage, Image* outputImage, unsigned int colorTableSize) :
        inputImage(inputImage), outputImage(outputImage), colorTableSize(colorTableSize)
    {
    }
    

    void init()
    {
        colorTable.resize(3 * colorTableSize);
        for(int i = 0; i < colorTableSize; i++)
        {   
            colorTable[3*i] = (i * 255) / colorTableSize ;
            colorTable[3*i+1] = (i * 255) / colorTableSize ;
            colorTable[3*i+2] = (i * 255) / colorTableSize;
        }

        acc.resize(3*colorTableSize);
        counter.resize(colorTableSize);
    }

    void iterate()
    {
        for(int i = 0; i < colorTableSize; i++)
            counter[i] = 0;
        for(int i = 0; i < 3*colorTableSize; i++)
            acc[i] = 0;

        for(int y = 0; y < inputImage->details.height; y++)
        for(int x = 0; x < inputImage->details.width; x++)
        {
            unsigned char r =inputImage->data.data()[3*(y * inputImage->details.width + x) + 0];
            unsigned char g = inputImage->data.data()[3*(y * inputImage->details.width + x) + 1];
            unsigned char b = inputImage->data.data()[3*(y * inputImage->details.width + x) + 2];
            int i = colorize(r,g,b);

            acc[3*i + 0] += r;
            acc[3*i + 1] += g;
            acc[3*i + 2] += b;
            counter[i] += 1;                             
        }
        for(int i = 0; i < colorTableSize; i++)
        {
            if(counter[i] == 0) counter[i] = 1;

            colorTable[3*i + 0] = acc[3*i + 0] / counter[i];
            colorTable[3*i + 1] = acc[3*i + 1] / counter[i];
            colorTable[3*i + 2] = acc[3*i + 2] / counter[i];
        }
    }

    void finalize()
    {

        for(int y = 0; y < inputImage->details.height; y++)
        for(int x = 0; x < inputImage->details.width; x++)
        {
            unsigned char r =inputImage->data.data()[3*(y * inputImage->details.width + x) + 0];
            unsigned char g = inputImage->data.data()[3*(y * inputImage->details.width + x) + 1];
            unsigned char b = inputImage->data.data()[3*(y * inputImage->details.width + x) + 2];
            int i = colorize(r,g,b);

            outputImage->data.data()[3*(y * outputImage->details.width + x) + 0] = colorTable[3*i + 0];
            outputImage->data.data()[3*(y * outputImage->details.width + x) + 1] = colorTable[3*i + 1];
            outputImage->data.data()[3*(y * outputImage->details.width + x) + 2] = colorTable[3*i + 2];
        }
    }

private:
    unsigned int colorize(unsigned char r, unsigned char g, unsigned char b)
    {
        unsigned int bestIndex = 0;
        float bestValue = std::numeric_limits<float>::max();

        for(unsigned int i = 0; i < colorTableSize; i++)
        {
            float l2 = (1.0 * colorTable[3*i + 0] - r)*(1.0 * colorTable[3*i + 0] - r) + (1.0 * colorTable[3*i + 1] - g)*(1.0 * colorTable[3*i + 1] - g) + (1.0 * colorTable[3*i + 2] - b)*(1.0 * colorTable[3*i + 2] - b);
            if(l2<bestValue)
            {
                bestIndex = i;
                bestValue = l2;
            }
        }

        return bestIndex;
    }
};

#endif // CPU_QUANTIZATION_H