__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

uint colorize(uint colorTableSize, __global uchar* colorTable, uint4 color)
{
    uint bestIndex = 0;
    float bestValue = FLT_MAX;

    for(uint i = 0; i < colorTableSize; i++)
    {
        float value = (colorTable[3*i] - color.x) * (colorTable[3*i] - color.x) +
                      (colorTable[3*i + 1] - color.y) * (colorTable[3*i + 1] - color.y) +
                      (colorTable[3*i + 2] - color.z) * (colorTable[3*i + 2] - color.z); 
        if(value < bestValue)
        {
            bestValue = value;
            bestIndex = i;
        }
    }

    return bestIndex;
}

__kernel void quantize(
        __read_only image2d_t image,
        uint colorTableSize,
        __global uchar* colorTable,
        __write_only image2d_t output
)
{

    const int2 dim = get_image_dim(image);
    const int2 cord;
    cord.x = get_global_id(0);
    cord.y = get_global_id(1);

    if(cord.x >= dim.x || cord.y >= dim.y)
    {
        return;
    }  

    uint4 px = read_imageui(image, sampler, cord);
    uint index = colorize(colorTableSize, colorTable, px);
    uint4 color = (uint4)(colorTable[3*index], colorTable[3*index+1], colorTable[3*index+2], 255);
    
    write_imageui(output, cord, color);
}

__kernel void accumulate(
        __read_only image2d_t image,
        uint colorTableSize,
        __global uchar* colorTable,
        __global uint* partialSums,
        __local uint* localSums
    )
{
    const int2 dim = get_image_dim(image);
    const int2 cord = (int2)(get_global_id(0), get_global_id(1));
    if(cord.x >= dim.x || cord.y >= dim.y)
        return;
    const uint2 localId = (uint2)(get_local_id(0), get_local_id(1));
    const uint2 groupId = (uint2)(get_group_id(0), get_group_id(1));
    const uint2 groups = (uint2)(get_num_groups(0), get_num_groups(1));
    const uint2 groupSize = (uint2)(get_local_size(0), get_local_size(1));

    uint4 px = read_imageui(image, sampler, cord);
    uint bestColorIndex = colorize(colorTableSize, colorTable, px);

    for(uint colorTableIndex = 0; colorTableIndex < colorTableSize; colorTableIndex++ )
    {
        uint localSumsIndex = localId.x + groupSize.x * localId.y;
        localSums[4*localSumsIndex] = (bestColorIndex == colorTableIndex) ? 1 : 0; // counter
        localSums[4*localSumsIndex + 1] = (bestColorIndex == colorTableIndex) ? px.x : 0; //r 
        localSums[4*localSumsIndex + 2] = (bestColorIndex == colorTableIndex) ? px.y : 0; //g
        localSums[4*localSumsIndex + 3] = (bestColorIndex == colorTableIndex) ? px.z : 0; //b

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int stride = (groupSize.x*groupSize.y) / 2; stride > 0; stride /= 2)
        {
            if(localSumsIndex < stride)
            {
                localSums[4*localSumsIndex] += localSums[4*(localSumsIndex + stride)];
                localSums[4*localSumsIndex+1] += localSums[4*(localSumsIndex + stride) + 1];
                localSums[4*localSumsIndex+2] += localSums[4*(localSumsIndex + stride) + 2];
                localSums[4*localSumsIndex+3] += localSums[4*(localSumsIndex + stride) + 3];
            }    
            barrier(CLK_LOCAL_MEM_FENCE);
        }


        if(localSumsIndex == 0)
        {
            uint groupindex = groupId.x + groupId.y * groups.x;
            partialSums[4*(colorTableSize * groupindex + colorTableIndex) + 0] =  localSums[4*localSumsIndex + 0];
            partialSums[4*(colorTableSize * groupindex + colorTableIndex) + 1] =  localSums[4*localSumsIndex + 1];
            partialSums[4*(colorTableSize * groupindex + colorTableIndex) + 2] =  localSums[4*localSumsIndex + 2];
            partialSums[4*(colorTableSize * groupindex + colorTableIndex) + 3] =  localSums[4*localSumsIndex + 3];
        }
    }

}

__kernel void partition(
        uint colorTableSize,
        __global uchar* colorTable,
        __global uint* partialSums,
        uint accGroupsNumber,
        uint accGroupsX
    )
{
    uint id = get_global_id(0);
    if(id >= colorTableSize)
        return; 

    uint counter = 0;
    uint racc = 0;
    uint gacc = 0;
    uint bacc = 0;

    for(int accGroup = 0; accGroup < accGroupsNumber; accGroup++)
    {
        counter += partialSums[4*(accGroup * colorTableSize + id) + 0]; 
        racc    += partialSums[4*(accGroup * colorTableSize + id) + 1]; 
        gacc    += partialSums[4*(accGroup * colorTableSize + id) + 2]; 
        bacc    += partialSums[4*(accGroup * colorTableSize + id) + 3]; 
    }

    if(counter == 0)
        counter = 1;
    colorTable[3*id    ] = racc / counter;
    colorTable[3*id + 1] = gacc / counter;
    colorTable[3*id + 2] = bacc / counter;
}