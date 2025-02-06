## Getting Started

1. **Prerequisites:**  
   - Julia
   - A MPI library
   - Required Julia packages (as specified in `Project.toml`)

2. **Running an Example:**  
   To run an example simulation for a spin chain with drive, long-ranged competing Ising interactions and incoherent decay, execute:
   ```sh
   julia spin_chain_demo.jl
   ```
   This simulation can be completed more efficiently by leveraging multiple processes. To do this, execute:
      ```sh
   mpirun -np X julia spin_chain_demo.jl
   ```
      where X is chosen the number of MPI workers.
   Refer to the comments in the file above for further explanation on problem and simulation setup, key functions, and required (hyper)parameters.
