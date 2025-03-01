include("../tVMPOMC.jl")
using .tVMPOMC
using MPI


mpi_cache = set_mpi()


#==========================================================
Physical System Parameters
==========================================================#
# Interaction parameters
const J1 = 0.5    # Primary interaction strength (ferromagnetic)
const J2 = -1.0   # Secondary interaction strength (antiferromagnetic)
const α1 = 3.0    # Power-law decay rate for J1 (r^-α1)
const α2 = 6.0    # Power-law decay rate for J2 (r^-α2)
const ising_int = "CompetingIsing" #Ising interaction type

# Remaining parameters
const hx = 0.5    # Transverse field strength (x-direction)
const hz = 0.0    # Longitudinal field strength (z-direction)
const γ = 1.0     # Spin decay rate
const N = 10      # Number of spins in chain

#==========================================================
Numerical Method Hyperparameters
==========================================================#
# MPO and Monte Carlo hyperparameters
const χ = 6       # Bond dimension (controls accuracy)
const uc_size = 1 # Unit cell size for translation symmetry
const N_MC = 1000 # Number of Monte Carlo samples
const burn_in = 2 # Monte Carlo equilibration steps
const sweeps = 2  # Samples between measurements

# Time evolution hyperparameters
const T = 2.0     # Total simulation time
const τ = 1e-8    # Initial time step
const max_τ = 0.1 # Maximum allowed time step

# Regularization hyperparameters
const ϵ_shift = 1e-9  # Numerical stability regularization
const ϵ_SNR = 1e-3    # Signal-to-noise ratio threshold
const ϵ_tol = 0.05    # SVD truncation tolerance

params = Parameters(N,χ,0,0,0,J1,J2,hx,hz,γ,0,α1,α2,uc_size)


#Define one-body Lindbladian operator:
l1 = make_one_body_Lindbladian(hx*sx+hz*sz, sqrt(γ)*sm)

#Define initial MPO:
mpo = MPO("-y", params, mpi_cache)

#Define sampler and optimizer:
sampler = MetropolisSampler(N_MC, burn_in, sweeps, params)
optimizer = TDVP(sampler, mpo, l1, τ, ϵ_shift, ϵ_SNR, ϵ_tol, params, ising_int)
NormalizeMPO!(params, optimizer)

#Save parameters to file:
dir = "Long_range_Ising_decay"
mkdir(dir)
cd(dir)


if mpi_cache.rank == 0
    
    #Save parameters to parameter file:
    list_of_parameters = open("Ising_decay.params", "w")
    redirect_stdout(list_of_parameters)
    display(params)
    display(sampler)
    display(optimizer)
    close(list_of_parameters)

    mx, my, mz = measure_magnetizations(params, mpo)

    list_of_obs = open("magnetizations.out", "a")
    println(list_of_obs, mx, ",", my, ",", mz)
    close(list_of_obs)

    list_of_times = open("times.out", "a")
    println(list_of_times, 0.0)
    close(list_of_times)
end

# Synchronize all MPI processes before proceeding:
MPI.Barrier(mpi_cache.comm)

# For non-root processes, change the working directory:
if mpi_cache.rank != 0
    cd(dir)
end

# Broadcast shared parameters from the root process to all other processes:
MPI.bcast(optimizer.τ, 0, mpi_cache.comm)
MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)

# Initialize simulation time:
current_time = 0.0

# Main simulation loop: run until the simulation time reaches target T:
while current_time<T

    # Update the optimizer state using an adaptive time-stepping method:
    AdaptiveHeunStepCapped!(max_τ, optimizer, mpi_cache)

    # Only the root process handles measurement output:
    if mpi_cache.rank == 0

        mx, my, mz = measure_magnetizations(params, optimizer.mpo)

        list_of_obs = open("magnetizations.out", "a")
        println(list_of_obs, mx, ",", my, ",", mz)
        close(list_of_obs)

        global current_time += optimizer.τ
        #list_of_times = open("times.out", "a")
        println(open("times.out", "a"), current_time)
        #close(list_of_times)
    end
end

MPI.Barrier(mpi_cache.comm)

error("Exiting")
