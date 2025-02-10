# t-VMPOMC: time-dependent variational MPO Monte Carlo

1. **Overview:**
t-VMPOMC is a Julia-based method for efficient and scalable simulation of the dynamics of open quantum lattices. It relies on solving the variational equations of motion efficiently by means of Monte Carlo, employing compact matrix product operator (MPO) trial states for the many-body density matrix.

2. **Core functionalities:**

3. **Prerequisites:**  
   - Julia
   - A MPI library
   - Required Julia packages (as specified in `Project.toml`)
  
4. **Installation:**
   Simply clone the repository above after making sure you installed the above prerequisites.
   
5. **Example simulation:**  
   To run an example simulation for a spin chain with drive, long-ranged competing Ising interactions and incoherent decay, execute:
   ```sh
   julia spin_chain_demo.jl
   ```
   The above measures the dynamics of the site-averaged mangetizations of the spin chain, saving the measurements to `magentizations.out`. For the set parameter values, the simulation should be completed within one minute on a modern PC.
   This simulation can be completed more efficiently by leveraging multiple parallel processes. To do this, execute:
      ```sh
   mpirun -np X julia spin_chain_demo.jl
   ```
      where X is a chosen the number of MPI workers.
   Refer to the comments in the file above for further explanation on problem and simulation setup, key functions, and required (hyper)parameters.

6. **Example cluster simulation:** 
