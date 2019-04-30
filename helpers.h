#ifndef HELPERS_H
#define HELPERS_H

inline size_t up(size_t v, size_t m = 32)
{
    return v + m - (v % m);
}

void informAboutStatus(int ret, const char* function, const char* file, int line)
{
    if(ret != 0)
        std::cout<<"RETURNED "<<ret<<" AT "<<file<<":"<<line<<" IN "<<function<<std::endl;
}

#define trace(ret) informAboutStatus(ret, __FUNCTION__, __FILE__, __LINE__)
#endif // HELPERS_H