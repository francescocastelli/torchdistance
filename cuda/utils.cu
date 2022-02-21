#include "utils.cuh"

TimingGPU::TimingGPU() { timer = std::make_unique<TimerGPU>(); }

void TimingGPU::StartCounter()
{
    cudaEventCreate(&((*timer).start));
    cudaEventCreate(&((*timer).stop));
    cudaEventRecord((*timer).start,0);
}

void TimingGPU::StartCounterFlags()
{
    int eventflags = cudaEventBlockingSync;

    cudaEventCreateWithFlags(&((*timer).start),eventflags);
    cudaEventCreateWithFlags(&((*timer).stop),eventflags);
    cudaEventRecord((*timer).start,0);
}

// Gets the counter in ms
float TimingGPU::GetCounter()
{
    float time;
    cudaEventRecord((*timer).stop, 0);
    cudaEventSynchronize((*timer).stop);
    cudaEventElapsedTime(&time,(*timer).start,(*timer).stop);
    return time;
}
