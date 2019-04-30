#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <functional>

#include "image.h"
#include "Quantization.h"
#include "KMeansCPUQuantization.h"
#include "KMeansGPUQuantization.h"

void measure(std::string title, std::function<void()> f)
{
    auto start = std::chrono::high_resolution_clock::now();
       
    f();
       
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    std::cout<< std::setw(32)<<title<<" : " << std::setw(13)<<std::fixed<<std::setprecision(2)<<elapsed.count() << " ms;"<< std::endl;
}

int main(int argc, char** argv)
{

    std::string inputFilename = "input1.png";
    std::string outputFilename = "output1.png";
    unsigned int colors = 32;
    unsigned int iterations = 3;

    if(argc >= 2)
        inputFilename = argv[1];
    if(argc >= 3)
        outputFilename = argv[2];
    if(argc >= 4)
        colors = atoi(argv[3]);
    if(argc >= 5)
        iterations = atoi(argv[4]);

    Image inputImage = readImage(inputFilename);
    Image outputImage = createBlankImage(inputImage.details.width, inputImage.details.height);


    std::vector<std::pair<std::string, Quantization*> > quantizers;

    quantizers.push_back(std::make_pair("CPU", new KMeansCPUQuantization(&inputImage, &outputImage, colors))); 
    quantizers.push_back(std::make_pair("atomic add GPU", new KMeansGPUQuantization(&inputImage, &outputImage, colors, "atomicAddKernel.cl"))); 
    quantizers.push_back(std::make_pair("parallel reduction GPU", new KMeansGPUQuantization(&inputImage, &outputImage, colors, "parallelReductionKernel.cl"))); 

    for(int q = 0; q < quantizers.size(); q++)
    {
        measure(quantizers[q].first + "/total    ", [&quantizers, &q, &iterations](){

            measure(quantizers[q].first+"/init     ", [&quantizers, &q](){
                    quantizers[q].second->init(); 
            });
            for(int i = 0; i < iterations; i++)
            {
                measure(quantizers[q].first+"/iteration", [&quantizers, &q](){
                        quantizers[q].second->iterate();
                });
            }

            measure(quantizers[q].first+"/finalize ", [&quantizers, &q](){
                    quantizers[q].second->finalize();
            });
        });

        delete quantizers[q].second;
    }

    writeImage(outputFilename, outputImage);

    return 0;
};
