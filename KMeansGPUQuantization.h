#ifndef KMEANS_GPU_QUANTIZATION
#define KMEANS_GPU_QUANTIZATION

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include "helpers.h"
#include "image.h"

class KMeansGPUQuantization : public Quantization
{
private:
    Image* inputImage;
    Image* outputImage;
    std::string kernelFilename;
    unsigned int colorTableSize;
    unsigned int localWorkSizeX;
    unsigned int localWorkSizeY;
    unsigned int accGroupsX;
    unsigned int accGroupsY;
    unsigned int accGroupsNumber;
    cl_platform_id platform_id;
    cl_device_id device_id;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_int ret;
    cl_kernel accKernel;
    cl_kernel partKernel;
    cl_kernel quantKernel;

    cl_mem inputClImage;
    cl_mem outputClImage;
    cl_mem  partialSumsClBuffer; 
    cl_mem  colorTableClBuffer;

    size_t imageOrigin[3];
    size_t imageRegion[3];
    size_t acc_local_work_size[3];
    size_t acc_global_work_size[3];
    size_t part_local_work_size[3];
    size_t part_global_work_size[3];
    size_t quant_local_work_size[3];
    size_t quant_global_work_size[3];
    cl_image_format clImageFormat;
    cl_image_desc clImageDesc;

    std::vector<unsigned char> colorTable;
    std::vector<unsigned int> partialSums;
public:
    KMeansGPUQuantization(Image* inputImage, Image* outputImage, unsigned int colorTableSize, std::string kernelFilename = "parallelReductionKernel.cl", 
        unsigned int localWorkSizeX = 32, unsigned int localWorkSizeY = 32) :
        inputImage(inputImage), outputImage(outputImage), colorTableSize(colorTableSize), kernelFilename(kernelFilename), localWorkSizeX(localWorkSizeX), localWorkSizeY(localWorkSizeY)
    {
    }

    ~KMeansGPUQuantization()
    {
        clFlush(command_queue);
        clFinish(command_queue);
        clReleaseKernel(accKernel);
        clReleaseKernel(partKernel);
        clReleaseKernel(quantKernel);
        clReleaseMemObject(colorTableClBuffer);
        clReleaseMemObject(partialSumsClBuffer);
        clReleaseMemObject(outputClImage);
        clReleaseMemObject(inputClImage);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
    }

    void init()
    {
        accGroupsX = up(inputImage->details.width, localWorkSizeX) / localWorkSizeX;
        accGroupsY = up(inputImage->details.height, localWorkSizeY) / localWorkSizeY;
        accGroupsNumber = accGroupsX * accGroupsY;

        initOpenCL();
        buildKernels();
        createImageObjects();
        createBufferObjects();
        clFinish(command_queue);
        setupKernels();
        clFlush(command_queue);
        clFinish(command_queue);
    }

    void iterate()
    {
        iterateAccumulation();
        iteratePartition();
        clFlush(command_queue);
        clFinish(command_queue);
    };


    void finalize()
    {
        quantizeImage();
        getQuantizedImageFromGPU();
        getColorTableFromGPU();
        clFlush(command_queue);
        clFinish(command_queue);
    };

private: 

    void initOpenCL()
    {
        ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms); trace(ret);
        ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices); trace(ret);
        context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret); trace(ret);
        command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret); trace(ret);
    }

    void buildKernels()
    {
        std::vector<std::string> sources;
        for(auto file : {kernelFilename})
            sources.push_back(readFile(file));
        auto [numberOfFiles, strings, lengths] = prepareSourcesForCL(sources);

        program = clCreateProgramWithSource(context, numberOfFiles, strings.data(), lengths.data(), &ret); trace(ret);
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL); trace(ret);
        if(ret != 0)
        {
            size_t len;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            std::string logs;
            logs.resize(len);
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, logs.data(), NULL);
            std::cout<<"Kernel compilation error: "<<std::endl<<logs<<std::endl;
        }

        accKernel = clCreateKernel(program, "accumulate", &ret); trace(ret);
        partKernel = clCreateKernel(program, "partition", &ret); trace(ret);
        quantKernel = clCreateKernel(program, "quantize", &ret); trace(ret);
    }


void createImageObjects()
    {
        clImageFormat = {CL_RGBA, CL_UNSIGNED_INT8};
        clImageDesc = {
            CL_MEM_OBJECT_IMAGE2D,          /* cl_mem_object_type image_type, */
            inputImage->details.width,      /* size_t image_width; */
            inputImage->details.height,     /* size_t image_height; */
            1,                              /* size_t image_depth; */
            1,                              /* size_t image_array_size; */
            0,                              /* size_t image_row_pitch; */ // OpenCL will calculate this
            0,                              /* size_t image_slice_pitch; */ // as above
            0,                              /* cl_uint num_mip_levels; */
            0,                              /* cl_uint num_samples; */
            NULL                            /* cl_mem mem_object; */ 
        };
        inputClImage = clCreateImage(context, CL_MEM_READ_WRITE, &clImageFormat, &clImageDesc, inputImage->data.data(), &ret); trace(ret);
        outputClImage = clCreateImage(context, CL_MEM_READ_WRITE, &clImageFormat, &clImageDesc, outputImage->data.data(), &ret); trace(ret);

        imageOrigin[0] = 0;
        imageOrigin[1] = 0;
        imageOrigin[2] = 0;
        imageRegion[0] = inputImage->details.width;
        imageRegion[1] = inputImage->details.height;
        imageRegion[2] = 1;

        ret = clEnqueueWriteImage(command_queue,
                               inputClImage,
                               CL_TRUE,
                               imageOrigin, /*const size_t *origin,*/
                               imageRegion, /*const size_t *region,*/
                               0, /*size_t input_row_pitch,*/ // OpenCL will calculate this
                               0, /*size_t input_slice_pitch,*/ // as above
                               inputImage->data.data(), /*const void *ptr,*/
                               0, /*cl_uint num_events_in_wait_list,*/
                               NULL, /*const cl_event *event_wait_list,*/
                               NULL /*cl_event *event*/
                            ); trace(ret);

    }

    void createBufferObjects()
    {
        colorTable.resize(colorTableSize * 3);
        for(int i = 0; i < colorTableSize; i++)
        {   
            colorTable[3*i] = (i * 255) / colorTableSize ;
            colorTable[3*i+1] = (i * 255) / colorTableSize ;
            colorTable[3*i+2] = (i * 255) / colorTableSize;
        }

        colorTableClBuffer = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          sizeof(unsigned char)*colorTable.size(),
                          NULL,
                          &ret); trace(ret);

        ret = clEnqueueWriteBuffer(command_queue,
                               colorTableClBuffer, /*cl_mem buffer,*/
                               CL_TRUE, /*cl_bool blocking_write,*/
                               0, /* size_t offset, */
                               sizeof(unsigned char)*colorTable.size(), /*size_t size,*/
                               colorTable.data(), /*const void *ptr,*/
                               0, /*cl_uint num_events_in_wait_list,*/
                               NULL, /* const cl_event *event_wait_list, */
                               NULL /*cl_event *event */); trace(ret);

        partialSums.resize(4*accGroupsX *accGroupsY * colorTableSize);
        partialSumsClBuffer = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          sizeof(unsigned int)*partialSums.size(),
                          NULL,
                          &ret); trace(ret);
        ret = clEnqueueWriteBuffer(command_queue,
                               partialSumsClBuffer, /*cl_mem buffer,*/
                               CL_TRUE, /*cl_bool blocking_write,*/
                               0, /* size_t offset, */
                               sizeof(unsigned int)*partialSums.size(), /*size_t size,*/
                               partialSums.data(), /*const void *ptr,*/
                               0, /*cl_uint num_events_in_wait_list,*/
                               NULL, /* const cl_event *event_wait_list, */
                               NULL /*cl_event *event */); trace(ret);
    }


    void setupKernels()
    {
        acc_local_work_size[0] = localWorkSizeX;
        acc_local_work_size[1] = localWorkSizeY;
        acc_local_work_size[2] = 1;
        acc_global_work_size[0] = up(inputImage->details.width, localWorkSizeX);
        acc_global_work_size[1] = up(inputImage->details.height, localWorkSizeY);
        acc_global_work_size[2] = 1;
        clSetKernelArg(accKernel, 0, sizeof(cl_mem), (void *)&inputClImage);
        clSetKernelArg(accKernel, 1, sizeof(unsigned int), (void *)&colorTableSize);
        clSetKernelArg(accKernel, 2, sizeof(cl_mem), (void *)&colorTableClBuffer);
        clSetKernelArg(accKernel, 3, sizeof(cl_mem), (void *)&partialSumsClBuffer);
        clSetKernelArg(accKernel, 4, 4*localWorkSizeX*localWorkSizeY*sizeof(unsigned int), NULL);

        part_local_work_size[0] = localWorkSizeX;
        part_local_work_size[1] = 1; 
        part_local_work_size[2] = 1;
        part_global_work_size[0] = up(colorTableSize, localWorkSizeX);
        part_global_work_size[1] = 1;
        part_global_work_size[2] = 1;
        clSetKernelArg(partKernel, 0, sizeof(unsigned int), (void *)&colorTableSize);
        clSetKernelArg(partKernel, 1, sizeof(cl_mem), (void *)&colorTableClBuffer);
        clSetKernelArg(partKernel, 2, sizeof(cl_mem), (void *)&partialSumsClBuffer);
        clSetKernelArg(partKernel, 3, sizeof(unsigned int), (void *)&accGroupsNumber);
        clSetKernelArg(partKernel, 4, sizeof(unsigned int), (void *)&accGroupsX);
        
        quant_local_work_size[0] = localWorkSizeX;
        quant_local_work_size[1] = localWorkSizeY; 
        quant_local_work_size[2] = 1;
        quant_global_work_size[0] = up(inputImage->details.width, localWorkSizeX);
        quant_global_work_size[1] = up(inputImage->details.height, localWorkSizeY);
        quant_global_work_size[2] = 1;
        clSetKernelArg(quantKernel, 0, sizeof(cl_mem), (void *)&inputClImage);
        clSetKernelArg(quantKernel, 1, sizeof(unsigned int), (void *)&colorTableSize);
        clSetKernelArg(quantKernel, 2, sizeof(cl_mem), (void *)&colorTableClBuffer);
        clSetKernelArg(quantKernel, 3, sizeof(cl_mem), (void *)&outputClImage);
    }

    void iterateAccumulation()
    {
        ret = clEnqueueNDRangeKernel(command_queue, accKernel, 2, NULL, acc_global_work_size, acc_local_work_size, 0, NULL, NULL); trace(ret);
    }

    void iteratePartition()
    {

        ret = clEnqueueNDRangeKernel(command_queue, partKernel, 1, NULL, part_global_work_size, part_local_work_size, 0, NULL, NULL); trace(ret);
    }

    void quantizeImage()
    {
        ret = clEnqueueNDRangeKernel(command_queue, quantKernel, 2, NULL, quant_global_work_size, quant_local_work_size, 0, NULL, NULL); trace(ret);
    }

    void getQuantizedImageFromGPU()
    {
        ret = clEnqueueReadImage(command_queue, 
            outputClImage, 
            CL_TRUE, 
            imageOrigin, /*  const size_t *origin */
            imageRegion, /* const size_t *region, */
            0,  /* size_t input_row_pitch, */ //OpenCl Will calculate this
            0,  /* size_t input_slice_pitch, */
            outputImage->data.data(), /*  const void *ptr,*/
            0, /* cl_uint num_events_in_wait_list, */ 
            NULL, /* const cl_event *event_wait_list, */  
            NULL /*  cl_event *event */
        ); trace(ret);

    }

    void getColorTableFromGPU()
    {
        ret = clEnqueueReadBuffer(command_queue,
                               colorTableClBuffer, /*cl_mem buffer,*/
                               CL_TRUE, /*cl_bool blocking_write,*/
                               0, /* size_t offset, */
                               sizeof(unsigned char)*colorTable.size(), /*size_t size,*/
                               colorTable.data(), /*const void *ptr,*/
                               0, /*cl_uint num_events_in_wait_list,*/
                               NULL, /* const cl_event *event_wait_list, */
                               NULL /*cl_event *event */); trace(ret);
    }

    std::string readFile(std::string filename)
    {
            std::FILE *file = std::fopen(filename.c_str(), "r");
            if(!file)
                    std::cout<<"Problem with opening: "<<filename<<std::endl;

            std::string content;
            std::fseek(file, 0, SEEK_END);
            content.resize(std::ftell(file));
            std::rewind(file);
            std::fread(&content[0], sizeof(char), content.size(), file);
            std::fclose(file);
            return std::move(content);
    }

    struct OpenCLProgram
    {
        size_t file;
        std::vector<const char*> strings;
        std::vector<size_t> lengths;
    };

    OpenCLProgram prepareSourcesForCL(const std::vector<std::string>& sources)
    {
        std::vector<size_t> lengths(sources.size());
        std::vector<const char*> strings(sources.size());

        for(int i = 0; i < sources.size(); i++)
        {
            strings[i] = sources[i].c_str();
            lengths[i] = sources[i].size();
        }

        return {sources.size(), std::move(strings), std::move(lengths)};
    }
};

#endif // KMEANS_GPU_QUANTIZATION_H