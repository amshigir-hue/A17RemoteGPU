#include <metal_stdlib>
using namespace metal;

kernel void square_ids(
    device uint *output [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        output[id] = id * id;
    }
}