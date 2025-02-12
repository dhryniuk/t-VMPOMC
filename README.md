# t-VMPOMC: time-dependent variational MPO Monte Carlo

1. **Overview:**
t-VMPOMC is a Julia-based method for efficient and scalable simulation of the dynamics of open quantum lattices. It relies on solving the variational equations of motion efficiently by means of Monte Carlo, employing compact matrix product operator (MPO) trial states for the many-body density matrix.

2. **Core functionalities:**

3. **Prerequisites:**  
   - Julia
   - Required Julia packages (as specified in `Project.toml`)
   - A MPI library
  
4. **Installation:**
   Simply clone the repository above after making sure you installed the above prerequisites.
   
5. **Example simulation:**  
   To run an example simulation for a spin chain with drive, long-ranged competing Ising interactions and incoherent decay, execute:
   ```sh
   julia Examples/spin_chain_demo.jl
   ```
   The above measures the dynamics of the site-averaged mangetizations of the spin chain, saving the measurements to `magentizations.out`, with corresponding simulation times saved to `times.out`. For the set parameter values, the simulation should be completed within one minute on a modern PC.
   This simulation can be completed more efficiently when parallelized over multiple CPUs with MPI. To do this, execute:
      ```sh
   mpirun -np X julia Examples/spin_chain_demo.jl
   ```
      where X is a chosen the number of MPI workers.
   Refer to the comments in the file above for further explanation on problem and simulation setup, key functions, and required (hyper)parameters.

6. **Example cluster simulation:**
   The pair of scripts `submit_Heisenberg.sh` and `Heisenberg_spin_chain.jl` constitute a simple example cluster simulation that can be run on a Sun Grid Engine server, and can be used to directly reproduce the t-VMPOMC results from Fig. 3 in the paper. To run it, simple execute the bash script:
      ```sh
   sh submit_Heisenberg.sh
   ```
   
