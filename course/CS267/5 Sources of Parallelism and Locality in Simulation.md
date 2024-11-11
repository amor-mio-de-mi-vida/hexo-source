---
date: 2024-11-09 19:14:18
date modified: 2024-11-10 15:06:46
title: Sources of Parallelism and Locality in Simulation
tags:
  - course
categories:
  - cs267
---
Introduce you to certain patterns that come up over and over again in parallel computing. 

What you can parallelize?

How you can keep data together on one processor or in one part of the memory in order to minimize communication. 
<!-- more -->

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.45p7sjb1i.webp)

| Shared Memory                             | Distributed Memory                             | SIMD                                              |
| ----------------------------------------- | ---------------------------------------------- | ------------------------------------------------- |
| Processors execute own instruction stream | Processors execute own instruction stream      | One instruction stream (all run same instruction) |
| Communicate by reading/writing memory     | Communicate by sending messages                | Communicate through memory                        |
| Cost of a read/write is constant          | Message time depends on size, but not location | Assume unbounded \# of arithmetic units           |
### Parallelism and Locality in Simulation

Parallelism and data locality both critical to performance

- Recall that moving data is the most expensive operation

Real world problems have parallelism and locality:

- Many objects operate independently of others.

- Objects often depend much more on nearby than distant objects.

- Dependence on distant objects can often be simplified. (E.g. particles moving under gravity)

Scientific models may introduce more parallelism:

- When a continuous problem is discretized, time dependencies are generally limited to adjacent time steps. (Helps limit dependence to nearby objects)

- Far-field effects may be ignored or approximated in many cases.

Many problems exhibit parallelism at multiple levels

### Basic Kinds of Simulation

**Discrete event systems:**

- "Game of Life," Manufacturing systems, Finance, Circuits, Pacman, ...

**Particle systems**: 

- Billiard balls, Galaxies, Atoms, Circuits, Pinball ...

**Lumped variables depending on continuous parameters**

- aka Ordinary Differential Equations (ODEs),

- Structural mechanics, Chemical kinetics, Circuits, Star Wars: The Force Unleashed

**Continuous variables depending on continuous parameters**

- aka Partial Differential Equations (PDEs)

- Heat, Elasticity, Electrostatics, Finance, Circuits, Medical Image Analysis, Terminator3: Rise of the Machines

A given phenomenon can be modeled at multiple levels. Many simulations combine more than one of these techniques.

## Discrete Event Systems

### Definition

Systems are represented as:

- finite set of variables

- the set of all variable values at a given time is called the state.

- each variable is updated by computing a transition function depending on the other variables.

System may be:

- **Synchronous**: at each discrete time-step evaluate all transition functions; also called a state machine

- **Asynchronous**: transition functions are evaluated only if the inputs change, based on an "event" from another part of the system; also called event driven simulation.

### Parallelism in Game of Life

The simulation is synchronous

- use two copies of the grid (old and new), "ping-pong" between them without worrying about race conditions.

- the value of each new grid cell depends only on 9 cells (itself plus 8 neighbors) in old grid.

- simulation proceeds in time-steps -- each cell is updated at every step.

Easy to parallelize by dividing physical domain: *Domain Decomposition*

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.6bh37zai43.webp)

Locality is achieved by using large patches of the ocean

- Only boundary values from neighboring patches are needed

How to pick shapes of domains?

Minimizing communication on mesh = minimizing "surface to volume ratio" of partition. 

### Synchronous Circuit Simulation

Circuit is a graph made up of subcircuits connected by wires

- Component simulations need to interact if they share a wire

- Data structure is (irregular) graph of subcircuits.

- Parallel algorithm is timing-driven or synchronous: Evaluate all components at every timestep (determined by known circuit delay)

Graph partitioning assigns subgraphs to processors

- Determines parallelism and locality.

- Goal 1 is to evenly distribute subgraphs to nodes (load balance).

- Goal 2 is to minimize edge crossing (minimize communication).

- Easy for meshes, NP-hard in general, so we will approximate.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.77dknftkk4.webp)

### Asynchronous Simulation

Synchronous simulations may waste time:

- Simulates even when the inputs do not change

Asynchronous (event-driven) simulations update only when **event** arrives from another component:

- No global time steps, but individual events contain time stamp.

- Example: Game of life in loosely connected ponds (don't simulate empty ponds)

- Example: Circuit simulation with delays (events are gates changing)

- Example: Traffic simulation (events are cars changing lanes, etc.)

Asynchronous is more efficient, but harder to parallelize

- On distributed memory, events are naturally implemented as messages between processors (eg using MPI), but how do you know when to execute a "receive" ?

### Scheduling Asynchronous Circuit Simulation

**Conservative:**

- Only simulate up to (and including) the minimum time stamp of inputs.

- Need deadlock detection if there are cycles in graph

- Example: Pthor circuit simulator in Splash1 from Stanford.

**Speculative (or Optimistic):**

- Assume no new inputs will arrive and keep simulating.

- May need to backup if assumption wrong, using timestamps.

- Example: Timewarp \[D. Jefferson\], Parswec \[Wen, Yelick\].

Optimizing load balance and locality is difficult:

- Locality means putting tightly coupled subcircuit on one processor.

- Since "activate" part of circuit likely to be in a tightly coupled subcircuit, this may be bad for load balance.

### Summary of Discrete Event Simulations

Model of the world is discrete (Both time and space)

**Approaches**

Decompose domain, i.e., set of objects

Run each component ahead using

- Synchronous: communicate at end of each timestep

- Asynchronous: communicate on-demand

- Conservative scheduling -- wait for inputs (need deadlock detection)

- Speculative scheduling -- assume no inputs (roll back if necessary)

## Particle Systems

### Definition

A particle system has

- a finite number of particles

- moving in space according to Newton's Laws

- Time and positions are continuous

Examples

- stars in space with laws of gravity

- electron beam in semiconductor manufacturing

- atoms in a molecule with electrostatic forces

- neutrons in a fission reactor

- cars on a freeway with Newton's laws plus model of driver and engine

- balls in a pinball game

Reminder: many simulations combine techniques such as particle simulations with some discrete events.

### Forces in Particle Systems

External force

- ocean current in sharks and fish world

- externally imposed electric field in electron beam

Nearby force

- sharks attracted to eat nearby fish

- balls on a billiard table bounce off of each other 

- Van der Waals forces in fluid ($1/r^6$)

Far-field force

- fish attract other fish by gravity-like ($1/r^2$) force 

- gravity, electrostatics, radiosity in graphics

- forces governed by elliptic PDE

### Parallelism in External Forces

These are the simplest

The force on each particle is independent

Evenly distribute particles on processors

- Any distribution works

- Locality is not an issue

For each particle on processor, apply the external force

- Also called "map" (eg absolute value)

- May need to "reduce" (eg compute maximum) to compute time step

### Parallelism in Nearby Forces

Nearby forces require interaction and therefore communication.

Force may depend on other nearby particles:

- Example: collisions.

- simplest algorithm is $O(n^2)$: look at all pairs to see if they collide.

Usual parallel model is domain decomposition of physical region in which particles are located

- $O(n/p)$ particles per processor if evenly distributed.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.6m3x166opf.webp)

**Challenge 1: interactions of particles near processor boundary:**

- need to communicate particles near boundary to neighboring processors. (Region near boundary called "ghost zone" or "halo")

- Low surface to volume ratio means low communication. (Use squares, not slabs, to minimize ghost zone sizes)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.7lk0ecdkja.webp)


**Challenge 2: load imbalance, if particles cluster:**

To reduce load imbalance, divide space unevenly.

- Each region contains roughly equal number of particles.

- Quad-tree in 2D, oct-tree in 3D.

- May need to rebalance as particles move, hopefully seldom.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4uay69vhs1.webp)

### Parallelism in Far-Field Forces

Far-field forces involve all-to-all interaction and therefore communication.

Force depends on all other particles:

- Examples: gravity, protein folding

- Simplest algorithm is $O(n^2)$

- Just decomposing space does not help since every particle needs to "visit" every other particle.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.8vmxko6x67.webp)

Use more clever algorithms to reduce communication

Use more clever algorithms to beat $O(n^2)$

### Far-field Forces: Particle-Mesh Methods

Based on approximation:

- Superimpose a regular mesh.

- "Move" particles to nearest grid point.

Exploit fact that the far-field force satisfies a PDE that is easy to solve on a regular mesh:

- FFT, multigrid (described in future lectures)

- Cost drops to $O(n\log n)$ or $O(n)$ instead of $O(n^2)$

Accuracy depends on the fineness of the grid is and the uniformity of the particle distribution.

**Step 1**: Particles are moved to nearby mesh points (scatter)

**Step 2**: Solve mesh problem

**Step 3**: Forces are interpolated at particles from mesh points (gather)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.58hdx5ij7m.webp)

### Far-field forces: Tree Decomposition

Based on approximation.

- Forces from group of far-away particles "simplified" -- resembles a single large particle.

- Use tree; each node contains an approximation of descendants.

Also $O(n\log n)$ or $O(n)$ instead of $O(n^2)$

Several Algorithms

- Barnes-Hut.

- Fast multipole method (FMM) of Greengard/Rohklin

- Anderson's method

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.67xhabpuu3.webp)

### Summary of Particle Methods

Model contains discrete entities, namely, particles

Time is continuous -- must be discretized to solve

Simulation follows particles through time steps

- `Force `= `externel_force` + `nearby_force` + `far_field_force`

- All-pairs algorithm is simple, but inefficient, $O(n^2)$

- Particle-mesh methods approximates by moving particles to a regular mesh, where it is easier to compute forces

- Tree-based algorithms approximate by treating set of particles as a group, when far away.

May think of this as a special case of a "lumped" system

## Lumped Systems: ODEs

### Definition

Many systems are approximated by 

- System of "lumped" variables.

- Each depends on continuous parameter (usually time).

Example -- circuit:

- approximate as graph: wires are edges; nodes are connections between 2 or more wires; each edge has resistor, capacitor, inductor or voltage source.

- system is "lumped" because we are not computing the voltage/current at every point in space along a wire, just endpoints.

- Variables related by Ohm's Law, Kirchoff's Laws, etc.

Forms a system of ordinary differential equations (ODEs)

- Differentiated with respect to time

- Variant: ODEs with some constraints (also called DAEs, Differential Algebraic Equations)

### Circuit Example

State of the system is represented by 

- $v_n(t)$ node voltages

- $i_b(t)$ branch currents

- $v_b(t)$ branch voltages

Equations include

- Kirchoff's current

- Kirchoff's voltage

- Ohm's law

- Capacitance

- Inductance

$$\begin{align}
\left(\begin{matrix}
0 & A & 0 \\
A' & 0 & -I \\
0 & R & -I \\
0 & -I & C*d/dt\\
0 & L*d/dt & I \\
\end{matrix}\right) * 
\left(\begin{matrix}
v_n\\
i_b\\
v_b
\end{matrix}\right)=
\left(\begin{matrix}
0\\S\\0\\0\\0\\
\end{matrix}\right)
\end{align}$$

A is sparse matrix, representing connections in circuit

- One column per branch (edge), one row per node (vertex) with +1 and -1 in each column at rows indicating end points

Write as single large system of ODEs or DAEs

### Structural Analysis Example

- Variables are displacement of points in a building

- Newton's and Hook's (spring) laws apply.

- Static modeling: exert force and determine displacement.

- Dynamic modeling: apply continuous force (earthquake)

- Eigenvalue problem: do the resonant modes of the building match an earthquake?


### Solving ODEs

In these examples, and most others, the matrices are sparse: 

- i.e., most array elements are 0

- neither store nor compute on these 0's

- Sparse because each component only depends on a few others

Given a set of ODEs, two kinds of questions are:

- Compute the values of the variables at some time t (Explicit methods && Implicit methods)

- Compute modes of vibration (Eigenvalue problems)

### Solving ODEs: Explicit and Implicit Methods

Assume ODE is $x'(t)=f(x)=A*x(t)$, where $A$ is a sparse matrix

- Compute $x(i*dt)=x[i]$, at $i=0,1,2,...$

- ODE gives $x'(i*dt)=\text{slope}$, $x[i+1]=x[i]+dt*\text{slope}$

Explicit methods, e.g., (Forward) Euler's method.

- Approximate $x'(t)=A*x(t)$ by $(x[i+1]-x[i])/dt=A*x[i]$

- $x[i+1]=x[i]+dt*A*x[i]$, i.e. sparse matrix-vector multiplication.

Tradeoffs:

- Simple algorithm: sparse matrix vector multiply.

- Stability problems: May need to take very small time steps, especially if system is stiff (i.e. A has some large entries, so $x$ can change rapidly.)

Implicit method, e.g. Backward Euler solve:

- Approximate $x'(t)=A*x(t)$ by $(x[i+1]-x[i])/dt=A*x[i+1]$

- $(I-dt*A)*x[i+1]=x[i]$, i.e. we need to solve a sparse linear system of equations.

Tradeoffs:

- Larger timestep possible: especially for stiff problems

- More difficult algorithm: need to solve a sparse linear system of equations at each step

### Solving ODEs: Eigensolvers

Computing modes of vibration: finding eigenvalues and eigenvectors

- Seek solution of $d^2x(t)/dt^2=A*x(t)$ of form $x(t)=\sin(\omega*t)*x_0$ where $x_0$ is a constant vector

$\omega$ called the frequency of vibration

$x_0$ sometimes called a "mode shape"

- Plug in to get $-\omega^2*x_0=A*x_0$, so that $-\omega^2$ is an eigenvalue and $x_0$ is an eigenvector of A.

- Solution schemes reduce either to sparse-matrix multiplications, or solving sparse linear systems.

### Summary of ODE Methods

Explicit methods for ODEs need sparse-matrix-vector multiplication.

Implicit methods for ODEs need to solve linear systems.

Direct methods (Gaussian elimination)

- Called LU Decomposition, because we factor $A=L*U$

- Future Lectures will consider both dense and sparse cases.

- More complicated than sparse-matrix vector multiplication.

Iterative solvers

- Will discuss several of these in future (Jacobi, Successive over-relaxation (SOR), Conjugate Gradient (CG), Multigrid ... )

- Most have sparse-matrix-vector multiplication in kernel.

Eigenproblems

- Future lectures will discuss dense and sparse cases.

- Also depend on sparse-matrix-vector multiplication, direct methods.

### SpMV in Compressed Sparse Row (CSR) Format

SpMV: $y=y+A*x$, only store, do arithmetic, on nonzero entries CSR format is simplest one of many possible data structures for A

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.9dcz9bct3r.webp)

```
for each row i
	for k=ptr[i] to ptr[i+1]-1 do
		y[i]=y[i] + val[k]*x[ind[k]]
```

### Matrix Reordering via Graph Partitioning

"Ideal" matrix structure for parallelism: block diagonal

- p (number of processors) blocks, can all be computed locally

- If no non-zeros outside these blocks, no communication needed

Can we reorder the rows/columns to get close to this?

- Most nonzeros in diagonal blocks, few outside.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.45p7xr9b7.webp)

### Goals of Reordering

Performance goals

**balance load** (how is load measured?).

- Approx equal number of nonzeros (not necessarily rows)

**balance storage** (how much does each processor store?).

- Approx equal number of nonzeros

**minimize communication** (how much is communicated?).

- Minimize nonzeros outside diagonal blocks

- Related optimization criterion is to move nonzeros near diagonal

**improve register and cache reuse**

- Group nonzeros in small vertical blocks so source (x) elements loaded into cache or registers may be reused (temporal locality)

- Group nonzeros in small horizontal blocks so nearby source (x) elements in the cache may be used (spatial locality)

Other algorithms reorder rows/columns for other reasons

- Reduce \# nonzeros in matrix after Gaussian elimination

- Improve numerical stability

### Graph Partitioning and Sparse Matrices

- Relationship between matrix and graph

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.2krxmv7v9l.webp)

- Edges in the graph are nonzero in the matrix: here the matrix is symmetric (edges are unordered) and weights are equal (1)

- If divided over 3 processors, there are 14 nonzeros outside the diagonal blocks, which represent the 7 (bidirectional) edges

### Summary: Common Problems

**Load Balancing **

- Statically -- Graph partitioning (Discrete event simulation; Sparse matrix vector multiplication)

- Dynamically -- If load changed significantly during job

**Linear algebra**

- Solving linear systems (sparse and dense)

- Eigenvalue problems will use similar techniques

**Fast Particle Methods**

- $O(n\log n)$ instead of $O(n^2)$

## Partial Differential Equations PDEs

### Definition

Examples of such systems include

**Elliptic problems (steady state, global space dependence)**

- Electrostatic or Gravitational Potential: Potential (position)

**Hyperbolic problems (time dependent, local space dependence)**

- Sound waves: Pressure (position, time)

**Parabolic problems (time dependent, global space dependence)**

- Heat flow: Temperature (position, time)

- Diffusion: Concentration (position, time)

Global vs Local Dependence

- Global means either a lot of communication, or tiny time steps

- Local arises from finite wave speeds: limits communication

Many problems combine features of above

- Fluid flow: Velocity, Pressure, Density (position, time)

- Elasticity: Stress, Strain (position, time)

### Details of the Explicit Method for Heat

$$\frac{du(x,t)}{dt}=C*\frac{d^2u(x,t)}{dx^2}$$

**Discretize** time and space using explicit approach (forward Euler) to approximate time derivative:
$$\begin{align}\frac{u(x,t+\delta)-u(x,t)}{\delta}&=C\frac{\frac{u(x-h,t)-u(x,t)}{h}-\frac{u(x,t)-u(x+h,t)}{h}}{h}\\&=C\frac{u(x-h,t)-2*u(x,t)+u(x+h,t)}{h^2}\end{align}$$
Solve for $u(x,t+\delta)$:
$$u(x,t+\delta)=u(x,t)+C*\delta/h^2*(u(x-h,t)-2*u(x,t)+u(x+h,t))$$
Let $z=C*\delta/h^2$, simplify:
$$u(x,t+\delta)=z*u(x-h,t)+(1-2z)*u(x,t)+z*u(x+h,t)$$
Change variable $x$ to $j*h$, $t$ to $i*\delta$, and $u(x,t)$ to `u[j,i]`
```psysudo algorithm
for j = 1 to N-1
	u[j,i+1]=z*u[j-1,i]+(1-2*z)*u[j,i]+z*u[j+1,i]
```
This corresponds to **Matrix-vector-multiply** by T; **Combine nearest neighbors on grid**.

{% note primary %}
 `u[j,i+1]=z*u[j-1,i]+(1-2*z)*u[j,i]+z*u[j+1,i]` is same as `u[:i+1]=T*u[:i]`, where T is tridiagonal:
{% endnote %}   

$$\begin{align}
&T=\left(\begin{matrix}
1-2z&z&&&\\
z&1-2z&z&&\\
&z&1-2z&z&&\\
&&z&1-2z&z\\
&&&z&1-2z\\
\end{matrix}\right)=I-z*L,\\
&L=\left(\begin{matrix}
2&-1&&&\\
-1&2&-1&&\\
&-1&2&-1&&\\
&&-1&2&-1\\
&&&-1&2\\
\end{matrix}\right)
\end{align}$$
- L called Laplacian (in 1D)

- For a 2D mesh (5 point stencil) the Laplacian is pentadiagonal

### Parallelism in Explicit Method for PDEs

**Sparse matrix vector multiply, via Graph Partitioning**

**Partitioning the space (x) into p chunks**

- good load balance (assuming large umber of points relative to p)

- minimize communication (least dependence on data outside chunk)

**Generalize to**

- multiple dimensions.

- arbitrary graphs (=arbitrary sparse matrices).

**Explicit approach often used for hyperbolic equations**

- Finite wave speed, so only depend on nearest chunks

**Problem with explicit approach for heat (parabolic)**

- numerical instability

- solution blows up eventually if $z=C\delta/h^2>0.5$

- need to make the time step $\delta$ very small when h. is small: $\delta < 0.5*h^2/C$

### Implicit Solution of the Heat Equation

$$\frac{du(x,t)}{dt}=C*\frac{d^2u(x,t)}{dx^2}$$
**Discretize** time and space using explicit approach (forward Euler) to approximate time derivative:
$$\begin{align}\frac{u(x,t+\delta)-u(x,t)}{\delta}&=C\frac{(u(x-h,t+\delta)-u(x,t+\delta))/h-(u(x,t+\delta)-u(x+h,t+\delta))/h}{h}\\&=C\frac{u(x-h,t+\delta)-2*u(x,t+\delta)+u(x+h,t+\delta)}{h^2}\end{align}$$
$$u(x,t)=u(x,t+\delta)-C*\delta/h^2*(u(x-h,t+\delta)-2*u(x,t+\delta)+u(x+h,t+\delta))$$
Let $z=C*\delta/h^2$ and change variable $t$ to $i*\delta$, $x$ to $j*h$ and $u(x,t)$ to `u[j,i]`

`(I+z*L)*u[:i+1]=u[:i]`

Where I is identity and L is Laplacian as before.

### Relation of Poisson to Gravity, Electrostatics

Poisson equation arises in many problems

force on particle at `(x,y,z)` due to particle at 0 is $-(x,y,z)/r^3$, where $r=\sqrt{x^2+y^2+z^2}$

Force is also gradient of potential $V=-1/r=-(\frac{dV}{dx},\frac{dV}{dy},\frac{dV}{dz})=-\text{grad}V$

V satisfies Poisson'e equation.

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.9kg75pk644.webp)

### 2D implicit Method

- Similar to the 1D case, but the matrix L is now

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4n7qbv9u6f.webp)

- Multiplying by this matrix (as in the explicit case) is simply nearest neighbor computation on 2D grid.

- To solve this system, there are several techniques.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.67xhbcchu3.webp)

### Overview of Algorithms

Sorted in two orders (roughly):

- from slowest to fastest on sequential machines.

- from most general (works on any matrix) to most specialized (works on matrix "like" T)

| Algorithm                                      | Description                                                                                                                                    |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Dense LU                                       | Gaussian elimination; works on any N-by-N matrix.                                                                                              |
| Band LU                                        | Exploits the fact that T is nonzero only on $\sqrt{N}$ diagonals nearest main diagonal.                                                        |
| Jacobi                                         | Essentially does matrix-vector multiply by T in inner loop of iterative algorithm.                                                             |
| Explicit Inverse                               | Assume we want to solve many systems with T, so we can precompute and store `inv(T)` "for free", and just multiply by it (but still expensive) |
| Conjugate Gradient                             | Uses matrix-vector multiplication, like Jacobi, but exploits mathematical properties of T that Jacobi does not.                                |
| Red-Black SOR <br>(successive over-relaxation) | Variation of Jacobi that exploits yet different mathematical properties of T. Used in multigrid schemes.                                       |
| Sparse LU                                      | Gaussian elimination exploiting particular zero structure of T.                                                                                |
| FFT (Fast Fourier Transform)                   | Works only on matrices very like T.                                                                                                            |
| Multigrid                                      | Also works on matrix like T, that come from elliptic PDEs                                                                                      |
| Lower Bound                                    | Serial (time to print answer) parallel (time to combine N inputs)                                                                              |

Details in class notes and www.cs.berkeley.edu/~demmel/ma221.

### Summary of Approaches to Solving PDEs

As with ODEs, either explicit or implicit approaches are possible

**Explicit**, sparse matrix-vector multiplication

**Implicit**, sparse matrix solve at each step

- Direct solvers are hard (more on this later)

- Iterative solves turn into sparse matrix-vector multiplication (Graph partitioning)

Graph and sparse matrix correspondence: 

- Sparse matrix-vector multiplication is nearest neighbor "averaging" on the underlying mesh

Not all nearest neighbor computations have the same efficiency

- Depends on the mesh structure (nonzero structure) and the number of Flops per point.

### Comments on practical meshes

Regular 1D, 2D, 3D meshes (Important as building blocks for more complicated meshes)

Practical meshes are often irregular

- **Composite meshes**, consisting of multiple "bent" regular meshes joined at edges

- **Unstructured meshes**, with arbitrary mesh points and connectivities.

- **Adaptive meshes**, which change resolution during solution process to put computational effort where needed.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.8l03sm0lum.webp)

### Challenges of Irregular Meshes

**How to generate them in the first place**

- Start from geometric description of object

- Triangle, a 2D mesh partitioner by Jonathan Shewchuk

- 3D harder!

**How to partition them**

- ParMetis, a parallel graph partitioner

**How to design iterative solvers**

- PETSc, a Portable Extensible Toolkit for Scientific Computing

- Prometheus, a multigrid solver for finite element problems on irregular meshes

**How to design direct solvers**

- SuperLU, parallel sparse Gaussian elimination

**These are challenges to do sequentially, more so in parallel**

### Summary -- sources of parallelism and locality

Current attempts to categorize main "kernels" dominating simulation codes

**Structured grids**: including locally structured grids, as in AMR

**Unstructured grids**

**Spectral methods (Fast Fourier Transform)**

**Dense Linear Algebra**

**Sparse Linear Algebra**: Both explicit (SpMV) and implicit (solving)

**Particle Methods**

**Monte Carlo/Embarrassing Parallelism/Map Reduce**


