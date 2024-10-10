---
date created: 2024-09-29 14:09:00
date modified: 2024-10-10 23:19:22
title: survey
tags:
  - riscv
categories:
  - riscv_simulator
date: 2024-09-29 14:09:00
---
# Paper

Date: June 27, 2019.

title: A Survey of Computer Architecture Simulation Techniques and Tools

contribution:
- Providing an up-to-date survey of computer architecture simulation techniques and simulators.
- Categorizing, analyzing and comparing various computer architecture simulators, which can help the community to understand the use-cases of different simulation tools.
- Providing detailed characteristics and experimental error comparison of six modern x86 computer architecture simulators: gem5 [2], Multi2sim [3], MARSSx86 [4], PTLsim [5], Sniper [6], and ZSim [7].
- Reviewing the most important challenges for architecture simulators and the solutions that have been proposed to resolve those issues

classification:
- ==Functional simulators==
	A functional simulator implements the architecture only and focuses on achieving the same functionality of the modeled architecture.In other words, functional simulators behave like emulators (emulate the behavior of target’s instruction set architecture (ISA)). 
- ==Timing simulators==, also known as performance simulators, simulate the microarchitecture of processors
	- ==Cycle-level Simulators==: Cycle-level simulators simulate an architecture by imitating the operation of the simulated processor for each cycle.
	- ==Event-driven Simulators==: An event-driven simulator simulates a target based on events instead of cycles. Usually, they make use of event queues. Simulation jumps to the time when an event is scheduled, based on the event queues, instead of going through all cycles.
	- ==Interval Simulators==:  regular instruction flow through the pipeline can be broken down into sets of intervals based on miss events (cache misses, branch mis-predictions). Special purpose portions of architectural simulators, like branch predictors and memory system, can be used to simulate the miss events and find their exact timings. Then, these timings along with an analytical model are used to estimate the duration for every interval of instructions.

To simplify the development and reduce its complexity, often simulators decouple functional and timing (performance) simulation.
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/riscv/Pasted-image-20240929164942.13lreaynv4.webp)
### Timing-directed simulator
In this category, a functional simulator ==records the architectural state== (e.g. register and memory values) of the processor being simulated. The timing simulator, which has no idea of data values on its own, takes and uses these values from the functional simulator to perform a specific task when required. The functional model and the timing models interact heavily in this type of simulators as the timing model directs the functional model and the functional model feeds values to the timing model. This interaction makes this simulation model suitable for modeling architectures with dynamically changing functional behavior, such as multicore architectures. For example, for a load instruction the functional model computes the instruction’s effective address, and the timing model uses this address to determine if the load is causing a cache miss. The returned value from the cache or the memory, will eventually be read by the functional simulator. 

### Functional-first simulators 
==the functional simulator runs prior to the timing simulator and generates an instruction trace (a stream of instructions) that feeds the timing simulator at runtime.== In the case of conditional branches, the functional simulator always follows the correct path and it cannot simulate the behavior of branch predictors. If there is a mispredicted branch in the timing simulator’s pipeline, the functional simulator restores its previous state before the branch and continues along the mispredicted path. Later, the pipeline has to be flushed due to this mispredicted branch. Since the timing simulator always lags behind the functional simulator, there can be ordering problems while simulating more than one thread . For instance, the time at which the functional model reads a memory value in case of a load instruction can be different from the time when the timing model requests the same value, and this can result in reading different values. This problem can be resolved by a speculative functional-first simulation. In this technique, whenever a timing model detects that the data it reads is different from the data that the functional model has read, it asks the functional model to restore the processor’s state to the state before the load instruction and then it executes the load instruction with the correct data. As, timing and functional models run in parallel, there is an opportunity to exploit this parallelism for better performance of the simulator. This type of simulators has much better performance as compared to timing-directed simulators, because it is not required for the timing model to direct the functional model at every instruction or cycle as in timing directed simulators. SimWattch is an example of functional first simulators. SimWattch integrates Simics with Wattch. Wattch is based on SimpleScalar and simulates both power and performance.

### Timing-first simulators
Timing-First Simulators: In this approach, timing simulators run ahead of functional simulators. Timing simulators simulate the microarchitecture of a target processor at the cycle-level. Timing simulators usually use functional simulators for verification of functional execution of all instructions. The instruction is retired in case of a match between the architectural state of both the functional and the timing simulators. In case of a mismatch, the timing simulator recovers by flushing the pipeline and restarting the instruction fetch following the problematic instruction. As such, the timing simulator makes forward progress. If these recoveries happen frequently, they can impact the simulated system’s timing, and thus accuracy, depending on the depth of the simulated pipeline.


classification: 
### Trace-Driven simulators
Trace files are used as inputs to trace-driven simulators. These trace files are prerecorded streams of instructions executed by benchmarks with some fixed inputs. As benchmarks execute on real machines statistics including instruction opcodes, data addresses, branch target addresses, etc are recorded in a trace file. Trace-driven model makes the implementation of the simulator simple. Trace-driven simulators can be easily debugged because experimental results can be reproduced. The size of trace files can be huge, which poses limits on the total instruction count in each trace file and/or the number of trace files used at once, and may lead to a slower simulation time [33], [57]. Different trace sampling and trace reduction techniques [58], are used to resolve the problem of large size of trace files. Apart from this, these simulators usually do not model execution of mispeculated code, which can affect performance estimation results of structures such as branch predictors. To solve the problem of branch mispredictions, techniques like reconstruction of mispredicted path [59] are used.
Trace-driven models do not include the run-time changes in behavior of multi-threaded applications [60]. This becomes a more visible problem if trace-driven simulation is run for a simulated multiprocessor system that is different from the one that was used to collect the trace. Trace-driven simulation should be avoided for parallel and timing-dependent systems as emphasized by Goldschmidt et al. [61].
Shade [62] is a trace-driven instruction set simulator, supporting SPARC and MIPS systems. Shade is also used to generate traces. Simplescalar also has the capability to run simulations from trace files. Cheetah [63] is a trace-driven simulator that simulates different cache con- figurations. MASE [64] is another example of this type of simulators. It is very hard for trace driven simulators to model the run-time changes in the behavior of multi-threaded applications [60], [61]. However, lately, few research works have been put forward to efficiently use trace-driven simulators for multi-threaded workloads, [65], [66].

### Execution-Driven Simlators

==These simulators use binaries or executables of benchmarks for simulated target machines directly. ==These simulators can simulate miss peculated instructions unlike trace-driven simulators. ==However, they are complicated as compared to trace-driven simulators. ==

Often, users are interested in the performance of selected regions of code instead of entire benchmarks. The technique of direct/native execution can help in this respect. In direct execution, ==simulators only simulate particular portions of code (or regions of interest) of an application and execute the rest of the application directly on the host machine.== In this case, both the target and the host systems should have same instruction set architecture (ISA) to perform native execution. This technique is also referred as co-simulation. 

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/riscv/Pasted-image-20240929172303.eshuabueg.webp)
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/riscv/Pasted-image-20240929172315.175dc0t0d3.webp)
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/riscv/Pasted-image-20240929172327.45hnfj1z2g.webp)


## Challenges
### A. slow simulation
- sampled simulation
- statistical simulation
- parallel simulation
- FPGA accelerated simulation

### B.poor accuracy
Potentially, there can be three different types of errors in simulators: Modeling errors, specification errors and abstraction errors.

- ==Modeling errors occur when the desired functionality is not properly implemented or modeled in the simulator==. One example of modeling errors is when instructions are con- figured to take different latencies than the modeled target. Another example can be issuing instructions to reservation stations in an out-of-order manner. Modeling errors can be reduced by carefully designing and testing the modeled structures. Errors can be further reduced using proper design strategies and software engineering principles.
- ==specification errors result from to the lack of knowledge about the correct functionality of the target.== Specification errors can only be decreased if the target’s specifications documentation is accessible. If certain specifications of the real hardware are not known, writing microbenchmarks can help estimating some specifications. For example, one can estimate the size of the reservation station by writing and running a microbenchmark for different cases.
- ==abstraction errors occur when developers implement their design at a higher level of abstraction to tradeoff design details for a better speed, or to simplify their simulator’s implementation. ==To reduce abstraction errors, developers usually tradeoff speed; simulator writers can reduce abstraction errors by including more details in their simulation models. Today’s new technologies with faster hardware enable further reduction of abstraction errors. ==Another example of an abstraction error is not simulating incorrect speculative paths, which can reduce the accuracy of the simulator. ==


- SimpleScalar
- EduMIPS64
- RARS
- Logisim



## Questions

1. block design
2. Incorporating structural


literature review and plan
- you implement  (minimum variable product) 
- you will do if you have time
- you will not do it      ——— future time

instruction set practice
sample projects.

what others do 
try it out

