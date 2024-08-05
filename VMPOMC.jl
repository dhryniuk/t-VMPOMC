module VMPOMC

using LinearAlgebra
using Distributed
using TensorOperations
using SparseArrays
using ArnoldiMethod
using Random
using MPI
using Statistics



#Basic utilities:
include("utils/projector.jl")
include("utils/parameters.jl")
include("utils/workspace.jl")
include("utils/utils.jl")
include("utils/mpi.jl")
include("utils/ED_Ising.jl")

 
#MPS/MPO backend:
include("MPO/MPO.jl")
include("MPO/observables.jl")


#Monte Carlo samplers:
include("Samplers/MetropolisSampler.jl")


#Optimizers:
include("Optimisers/diagonal_operators.jl")
include("Optimisers/optimizer.jl")


#Optimizer routines:

include("Optimisers/MPO/StochasticTDVP.jl")
#include("Optimisers/MPO/ExactTDVP.jl")
include("Optimisers/MPO/Ising_interactions.jl")


include("Samplers/MPO_Metropolis.jl")

end