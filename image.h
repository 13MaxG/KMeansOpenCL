#ifndef IMAGE_H
#define IMAGE_H

#include <png.h>
#include <cstring>


struct Image
{
    png_image details;
    std::vector<unsigned char> data;
};


Image readImage(std::string filename)
{
    Image output;

    memset(&(output.details), 0, (sizeof output.details));
    output.details.version = PNG_IMAGE_VERSION;
    int ret;
    ret = png_image_begin_read_from_file(&(output.details), filename.c_str());
    output.details.format = PNG_FORMAT_RGBA;

    output.data.resize(PNG_IMAGE_SIZE(output.details));
    ret = png_image_finish_read(&(output.details), NULL, output.data.data(), 0/*row_stride*/, NULL/*colormap*/);
    
    return std::move(output);
}

void writeImage(std::string filename, Image &image)
{
    png_image_write_to_file(&(image.details), filename.c_str(), 0/*convert_to_8bit*/, image.data.data(), 0/*row_stride*/, NULL/*colormap*/);
}

Image createBlankImage(int width, int height)
{
    Image output;
    memset(&(output.details), 0, (sizeof output.details));
    output.details.version = PNG_IMAGE_VERSION;
    output.details.format = PNG_FORMAT_RGBA;
    output.details.width = width;
    output.details.height = height;
    output.data.resize(PNG_IMAGE_SIZE(output.details));

    return std::move(output);
}

#endif // IMAGE_H