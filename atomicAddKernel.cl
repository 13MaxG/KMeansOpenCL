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

    atomic_add(partialSums + 4*bestColorIndex + 0, 1);
    atomic_add(partialSums + 4*bestColorIndex + 1, px.x);
    atomic_add(partialSums + 4*bestColorIndex + 2, px.y);
    atomic_add(partialSums + 4*bestColorIndex + 3, px.z);
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

    counter = partialSums[4* id + 0]; 
    racc    = partialSums[4* id + 1]; 
    gacc    = partialSums[4* id + 2]; 
    bacc    = partialSums[4* id + 3]; 


    if(counter == 0)
        counter = 1;
    colorTable[3*id    ] = racc / counter;
    colorTable[3*id + 1] = gacc / counter;
    colorTable[3*id + 2] = bacc / counter;
    partialSums[4* id + 0] = 0; 
    partialSums[4* id + 1] = 0; 
    partialSums[4* id + 2] = 0; 
    partialSums[4* id + 3] = 0; 
}