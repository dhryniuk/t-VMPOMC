## Getting Started

1. **Prerequisites:**  
   - Julia
   - MPI
   - Required Julia packages (as specified in `Project.toml`)

2. **Running an Example:**  
   To run the spin chain example, execute:
   ```sh
   julia Example\ scripts/Spin_chain_with_competing_Ising_interactions.jl <N> <χ> <uc_size> <N_MC> <T> <ϵ_shift> <ϵ_SNR> <ϵ_tol> <max_τ>