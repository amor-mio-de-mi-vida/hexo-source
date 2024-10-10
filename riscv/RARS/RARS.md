---
date created: 2024-10-02 14:59:29
date modified: 2024-10-06 11:01:54
title: RARS
tags:
  - riscv
categories:
  - RARS
---
## Command Line Tool
```text
a -- assemble only, do not simulate
ae<n> -- terminate RARS with integer exit code <n> if an assemble error occurs.
ascii -- display memory or register contents interpreted as ASCII codes.
b -- brief - do not display/memory address along with contents.
d -- display RARS debugging statements
```


## Tools
### Timer Tool
Use this tool to simulate the Memory Mapped IO (MMIO) for a timing device allowing the program to utilize timer interrupts. While this tool is connected to the program it runs a clock (starting from time 0), storing the time in milliseconds. The time is stored as a 64 bit integer and can be accessed (using a lw instruction) at 0xFFFF0018 for the lower 32 bits and 0xFFFF001B for the upper 32 bits.

### Keyboard and Display MMIO Simulator


### Data Cache simulation Tool


### Bitmap Display


### Instruction Statistics


### Instruction Counter


### Digital Lab Sim


### BHT Simulator



### Memory Reference Visualization


### Instruction/Memory Dump


### Floating Point Representation



