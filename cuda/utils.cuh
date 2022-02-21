#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __TIMING_CUH__
#define __TIMING_CUH__

class TimingGPU
{
public:
        TimingGPU();

        void StartCounter();
        void StartCounterFlags();
        float GetCounter();

private:
	struct TimerGPU {
	    cudaEvent_t start;
	    cudaEvent_t stop;
	};

	std::unique_ptr<TimerGPU> timer;

}; // TimingCPU 

#endif
