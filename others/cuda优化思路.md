---
date: 2024-11-25 13:28:30
date modified: 2024-11-25 13:30:43
title: Welcome to use Hexo Theme Keep
tags:
  - Hexo
  - Keep
categories:
  - Hexo
---
1. memory coalescing，保证[内存](https://product.it168.com/list/b/0205_1.shtml)融合。因为global memory在CC为1.x上是按照half wrap进行访问读写的，而在2.x上是按照wrap进行访问读写的。在显存中，有多个存储器控制器，负责对显存的读写，因此，一定要注意存储器控制器的[负载均衡](https://product.it168.com/list/b/0462_1.shtml)问题。每一个存储器控制器所控制的那片显存中的地址空间称为一个分区。连续的256Byte数据位于同一个分区，相邻的另一组256Byte数据位于另一个分区。访问global memory就是要让所有的分区同时工作。合并访问就是要求同一half-wrap中的thread按照一定byte长度访问对齐的段。在1.0和1.1上，half-wrap中的第k个thread必须访问段里的第k个字，并且half-wrap访问的首地址必须是字长的16倍，这是因为1.0和1.1按照half-wrap进行访问global memory，如果访问的是32bit字，比如说一个float，那么half-wrap总共访问就需要16个float长，因此，每个half-wrap的访问首地址必须是字长的16倍。1.0和1.x只支持对32bit、64bit和128bit的合并访问，如果不能合并访问，就会串行16次。1.2和1.3改进了1.0和1.1的访问要求，引进了断长的概念，与1.0和1.1上的端对齐长度概念不同，支持8bit-段长32Byte、16bit-段长64Byte、32bit-64bit-128bit-段长128Byte的合并访问。对1.2和1.3而言，只要half-wrap访问的数据在同一段中，就是合并访问，不再像1.0和1.1那样，非要按照顺序一次访问才算合并访问。如果访问的数据首地址没有按照段长对齐，那么half-wrap的数据访问会分两次进行访问，多访问的数据会被丢弃掉。所以，下面的情况就很容易理解：对1.0和1.1，如果thread的ID与访问的数据地址不是顺序对应的，而是存在交叉访问，即：没有与段对齐，那么，就会16次串行访问，而对1.2和1.3来讲，会判断这half-wrap所访问的数据是不是在同一个128Byte的段上，如果是，则一次访问即可，否则，如果half-wrap访问地址连续，但横跨两个128Byte，则会产生两次 传输，一个64Byte，一个32Byte。当然，有时还要考虑wrap的ID的奇偶性。1.2和1.3放宽了对合并访问的条件，最快的情况下的带宽是最好的情况下的带宽的1/2，然而，如果half-wrap中的连续thread访问的显存地址相互间有一定的间隔时，性能就会灰常差。比如，half-wrap按列访问矩阵元素，如果thread的id访问`2*id`的地址空间数据，那么，半个wrap访问的数据刚好是128Byte，一次访问可以搞定，但是，有一半数据会丢失，所以，也表示浪费了带宽，这一点一定要注意。如果不是2倍，而是3倍、4倍，那么，有效带宽继续下降。在程序优化时，可以使用share memory来避免间隔访问显存。

2. bank conflict，bank冲突。先说一下，share memory在没有bank conflict情况下，访问速度是global和local的100倍呢，你懂的。类似global memory的分区，share memory进行了bank划分。如果half-wrap内的很多thread同时要求访问同一个bank，那么就是bank conflict，这时，硬件就会将这些访问请求划分为独立的请求，然后再执行访问。但是，如果half-wrap内所有thread都访问同一个bank，那么会产生一次broadcast广播，只需要一次就可以相应所有访问的请求。每个bank宽度长度为32bit。对于1.x来讲，一个SM中的share memory被划分为16个bank，而2.x是32个bank。1.x的bank conflict和2.x的bank conflict是不一样的。对1.x来讲，多个thread访问同一个bank，就会出现bank conflict，half-wrap内所有thread访问同一个bank除外。但是，对2.x来讲，多个thread访问同一个bank已经不再是bank conflict了。比如：

　　__shared__ char Sdata[32];  
  
  
  
　　char data = Sdata[BaseIndex+tid];

　　在1.x上属于bank conflict，因为，0~3thread访问同一个bank，4~7访问同一个bank，类推，这种情况属于4-way bank conflict。但是，对于2.x来讲，这种情况已经不是bank conflict了，以为2.x采用了broadcast机制，牛吧，哈哈。 这里要多看看矩阵乘积和矩阵转置例子中的share memory的使用，如何保证memory coalescing和避免bank conflict的。

3. texture memory是有cache的，但是，如果同一个wrap内的thread的访问地址很近的话，那么性能更高。

　　
  
4. 以下是要注意的：

　　(1)在2.x的CC上，L1 cache比texture cache具有更高的数据带宽。所以，看着使用哈。

　　(2)对global memory的访问，1.0和1.1的设备，容易造成memory uncoalescing，而1.2和1.3的设备，容易造成bandwidth waste。 而对2.x的设备而言，相比1.2和1.3，除了多了L1 cache，没有其他的特别之处。

　　(3）采用-maxrregcount=N阻止complier分配过多的register。

　　(4)occupancy是每个multiprocessor中active wrap的数目与可能active wrap的最大数目的比值。higher occupancy并不意味着higher performance，因为毕竟有一个点，超过这个点，再高的occupancy也不再提高性能了。

5. 影响occupancy的一个因素，就是register的使用量。比如，对于1.0和1.1的device来讲，每个multiprocessor最多有8192个register，而最多的simultaneous thread个数为768个，那么对于一个multiprocessor，如果occupancy达到100%的话，每个thread最多可以分配10个register。另外，如果在1.0和1.1上，一个kernel里面的一个block有128个thread，每个thread使用register个数为12，那么，occupancy为83%，这是因为一个block有128个thread，则，由于multiprocessor里面最大的simultaneous thread为768，根据这个数目计算，最多同时有6个active block，但是6个active block，就会导致总共thread个数为128*6*12个，严重超过了8192，所以不能为6，得为5，因为128*5<768, and 128*5*12<8192, 5是满足要求的最大的整数。如果一个kernel里面的一个block有256个thread，同样一个thread用12个register，那么occupancy为66%，因为active block为2。可以在编译选项里面加入--ptxas-options=-v查看kernel中每个thread使用register的数量。同时，NV提供了CUDA_Ocuppancy_calculator.xls作为occupancy计算的辅助工具。顺便说一下，对于1.2和1.3的device来讲，每个multiprocessor最多的simultaneous thread个数为1024个。

6. 为了隐藏由于register dependent寄存器依赖造成的访问延迟latency，最小要保证25%的occupancy，也就是说，对于1.x的device来讲，一个multiprocessor最少得发起192个thread。对于1.0和1.1来讲， occupancy为192/768=25%，达到要求，但是对于1.2和1.3而言，192/1024=18.75%，不过，也只能这样。对于2.x系列的device来讲，由于是dual-issue，一个multiprocessor最多发起simultaneous thread个数为1536个，所以，一个multiprocessor最少同时发起384个thread时，occupancy为384/1536=25%，又达到了25%。

 7. 对于block和thread的分配问题，有这么一个技巧，每个block里面的thread个数最好是32的倍数，因为，这样可以让计算效率更高，促进memory coalescing。其实，每个grid里面block的dimension维度和size数量，以及每个block里面的thread的dimension维度和size数量，都是很重要的。维度呢，采用合适的维度，可以更方便的将并行问题映射到[CUDA](http://cudazone.nvidia.cn/what-cuda/ "CUDA")架构上，但是，对性能不会有太大改进。所以，size才是最重要的，记住叻! 其实，访问延迟latency和occupancy占有率，都依赖于每个multiprocessor中的active wrap的数量，而active wrap的数量，又依赖于register和share memory的使用情况。首先，grid中block的数目要大于multiprocessor的数目，以保证每个multiprocessor里面最少有一个block在执行，而且，最好有几个active block，使得blocks不要等着__syncthreads()，而是占用了hardware。其次，block里面的thread的数目也很重要。对于1.0和1.1的设备来讲，如果一个kernel里面block的大小为512个thread，那么，occupancy为512/768=66%，并且一个multiprocessor中只有一个active block，然而，如果block里面的thread为256个thread，那么，768/256=3，是整数，因此，occupancy为100%，一个multiprocessor里面有3个active block。但是，记住了，higher occupancy don't mean better performance更高的占有率并不意味着更好的性能。还是刚才那个例子，100%的occupancy并不比66%的occupancy的性能高很多，因为，更低的occupancy使得thread可以有更多的register可以使用，而不至于不够用的register分配到local memory中，降低了变量存取访问速度。一般来讲啊，只要occupancy达到了50%，再通过提高occupancy来提高性能的可能性不是很大，不如去考虑如何register和share memory的使用。保证memory coalescing和防止bank conflict。记住如下几点：

　　(1)block里面thread个数最好为wrap大小的倍数，即：32的倍数。使得计算效率更高，保证memory coalescing。

　　(2)如果multiprocessor中有多个active block时，每个block里面的thread个数最好为64的倍数。

　　(3)当选择不同的block大小时，可以先确定block里面thread个数为128到256之间，然后再调整grid中block大小。

　　(4)如果是让问延迟latency造成程序性能下降时，考虑在一个block里面采用小block划分，不要在一个multiprocessor中分配一个很大的block，尽量分配好几个比较小的block，特别是程序中使用了__syncthreads()，这个函数是保证block里面所有wrap到这里集合，所以，block里面的thread越少越好，最好是一个wrap或者两个wrap，这样就可以减少__syncthreads()造成的访问延迟。

　　(5)如果如果一个block里面分配的register超过了multiprocessor的最大极限时，kernel的launch就会fail。

8. share memory的使用量也是影响occupancy的一个重要因子。thread与share memory的元素之间，没有必要是一对一的。一个线程可以一起负责处理share memory数组中的第一个、第二个以及第三个元素，都ok的。第一个thread处理share memory中的第一个元素，第二个thread负责处理第二个元素，类推如下，这种情况不是必须的，有时也没必要这么做。在代码里面，采用一个thread负责处理share memory数组中的多个元素的方法，是非常好的策略。这是因为如果share memory里面各个元素要进行相同的操作的话，比如乘以2，那么，这些操作可以被负责处理多个元素的一个thread一次搞定，分摊了thread处理share memory元素数量的成本费用。

　　9. 当上面那些high level级别的优化策略都检查使用过以后，就可以考虑low level级别的优化：instruction optimization指令集优化。这个也可以很好的提高性能的。指令集的优化，可以稍微总结如下：

　　(1)尽量使用shift operation位移运算来取代expensive昂贵的division除法和modulo取余运算，这里说的都是integer运算，float不行的。如果n是2幂数，(i/n)=(i>>log2(n)), (i%n)=(i&(n-1)). 其实，这只是一个量的问题，对于1.x的设备而言，如果一个kernel里面使用了十多个tens of这样的指令，就要考虑用位移运算来取代了;对于2.x的设备而言，如果一个kernel里面使用了20个这样的指令，也要考虑使用位移运算来取代除法和取余运算。其实，compiler有时会自动做这些转换的，如果n是2的幂数。

　　(2)reciprocal square root，对于平方根倒数1.0f/sqrtf(x)，编译器会采用rsqrtf(x)来取代，因为硬件做了很多优化。当然，对于double型的平方根倒数，就采用rsqrt(x)啦。呵呵，记住了。

　　(3)编译器有时会做一些指令转化。在要做计算的单精度浮点型常数后面，一定要加入f，否则，会被当做双精度浮点运算来计算，对于2.x以上的设备来讲，这一点很重要，记好了。

　　(4)如果追求速度speed，而不是精度precision，那么尽量使用fast math library。比如，__sinf(x)、__expf(x)比sinf(x)和expf(x)有更快的速度，但是，精度却差一些。如果是__sinf(x-2)则比sinf(x-2)的速度要快一个数量级，因为x-2运算用到了local memory，造成太高的访问延迟。当然，在compiler option中使用-use_fast_math可以让compiler强制将sinf(x)和expf(x)转化为__sinf(x)和__expf(x)进行计算。对于transcendental function超越函数，作用对象是单精度浮点型数据时，经常这么用，其他类型的数据，性能提升不大。

　　(5)对于2和10为底做指数运算，一定要采用exp2()或者expf2()以及exp10()或者expf10()，不要采用pow()和powf()，因为后者会消耗更多的register和instruction指令。 另外，exp2()、expf2()、exp10()、expf10()的性能和exp()以及expf()性能差不太多，当然比pow()和powf()要快10多倍呢。加好了哈。

　　(6)减少global memory的使用，尽量将global memory数据加载到share memory，再做访问。因为访问uncached的显存数据，需要400~600个clock cycle的[内存](https://product.pcpop.com/Memory/10734_1.html)延迟。

　　10. 下一个就是control flow了。一定要避免在同一个wrap里面发生different execution path。尽量减少if、swith、do、for、while等造成同一个wrap里面的thread产生diverge。因为，一旦有divergence，不同的execution path将会顺序的串行的执行一遍，严重影响了并行性。但是：

switch（threadIdx.x）  
  
  
  
{  
  
  
  
case 0：  
  
  
  
break;  
  
  
  
case 1:  
  
  
  
break;  
  
  
  
...  
  
  
  
case 31:  
  
  
  
break;  
  
  
  
}

　　上面这个例子，则不会发生divergence，因为控制条件刚好和wrap里面的thread相对应。

　　其实，有时，compiler会采用branch predication分支预测来打开loop循环或者优化if和switch语句， 这时，wrap就不会出现divergence了。在写code时，我们也可以自己采用#pragma uroll来打开loop循环。在使用branch predication时，所有指令都将会执行，其实，只有预测正确的真正的执行了，而预测错误的，其实就是thread，不会去读取该instruction的地址和数据，也根本不会写结果。其实，编译器做分制预测，是有条件的，只有分支条件下的指令instruction的个数小于等于某个阈值的时候，才会做分支预测branch predication。如果编译器觉得可能会产生多个divergent wrap，那么阈值为7，否则为4。(这里很不理解7和4是怎么来的)。

11. 在loop循环的counter，尽量用signed integer，不要用unsigned integer。比如：for(i = 0; i < n; i++) {out[i] = in[offset+stride*i];} 这里呢，stride*i可以会超过32位integer的范围，如果i被声明为unsigned，那么stride*i这个溢出语句就会阻止编译器做一些优化，比如strength reduction。相反，如果声明为signed，也没有溢出语句时，编译器会对很多地方做优化。所以，loop counter尽量设置为int，而不是unsigned int。

12. 在1.3及其以上的device上，才支持double-precision floating-point values，即：64位双精度浮点运算。当使用double时，在编译器选项里面添加：-arch=sm_13

13. 还有一点需要注意，如果A、B、C都是float，那么A+(B+C)并不一定等于(A+B)+C。

14. 先看下面两个语句：float a; a = a * 1.02;

　　对于1.2及其以下的device来讲，或者1.3及其以上device，但是没有打开支持double运算的选项，那么，由于不支持double，所以，1.02*a这个乘积是一个float;

　　对于1.3及其以上的device来讲，如果打开了支持double运算的选项，那么，a*1.02是一个double，而将乘积赋值给a，这个结果是float，所以，是先做了从float到double的promotion扩展，然后做了从double到float的truncation截取。

15. 多GPU编程。如果有p个GPU同时并行，那么，程序中就需要p个[CPU](https://product.it168.com/list/b/0217_1.shtml) threads。这些thread可以用OpenMP(小规模)或者MPI(大规模)进行管理。GPU之间的数据拷贝，必须通过[CPU](https://product.pcpop.com/CPU/10734_1.html)实现。对于OpenMP，是这样的：一个CPU thread将数据从对应的GPU中拷贝到host端的share memory region中，然后另一个CPU thread将数据从host端的share memory region拷贝到对应的GPU中。也就是说：OpenMP是通过share memory进行数据拷贝的。而对于MPI而言，数据是通过message passing进行传递的。一个CPU thread使用cudaMemcpy将数据从device拷贝到host，然后通过MPI_Sendrecv()，另一个CPU thread就使用cudaMemcpy将数据从host端拷贝到呃device端。编译选项，记着采用nvcc -Xcompiler /openmp或者nvcc -Xcompiler mpicc。