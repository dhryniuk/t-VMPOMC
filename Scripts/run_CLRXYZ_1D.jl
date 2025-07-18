include("../tVMPOMC.jl")
using .tVMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI
using Dates
using JLD

mpi_cache = set_mpi()

#Set parameters:
Jx_1= 0.9 #interaction strength
Jy_1= -1.0 #interaction strength
Jz_1= 1.1 #interaction strength
Jx_2= -0.6 #interaction strength
Jy_2= 0.5 #interaction strength
Jz_2= -0.4 #interaction strength
J1= 0.0 #interaction strength
J2= 0.0 #interaction strength
hx= 0.5 #transverse field strength
hz= 0.0 #longitudinal field strength
γ = 1.0 #spin decay rate
N = parse(Int64,ARGS[1]) #number of spins
α1 = 3.0
α2 = 6.0
γ_d = 0

#Set hyperparameters:
χ = parse(Int64,ARGS[2]) #MPO bond dimension
uc_size = parse(Int64,ARGS[3])
N_MC = parse(Int64,ARGS[4]) #number of Monte Carlo samples
burn_in = 2 #Monte Carlo burn-in
sweeps = 2
T = parse(Float64,ARGS[5])
ϵ_shift = parse(Float64,ARGS[6])
ϵ_SNR = parse(Float64,ARGS[7])
ϵ_tol = parse(Float64,ARGS[8])
max_τ = parse(Float64,ARGS[9])
#ising_int = "Ising"
ising_int = "CompetingIsing"
#τ = 0.01
τ = 10^(-8)

params = Parameters(N,χ,Jx_1,Jy_1,Jz_1,J1,J2,hx,hz,γ,γ_d,α1,α2,uc_size)

#Define one-body Lindbladian operator:
l1 = make_one_body_Lindbladian(hx*sx+hz*sz, sqrt(γ)*sm)
l2_1 = Jx_1*make_two_body_Lindblad_Hamiltonian(sx, sx) + Jy_1*make_two_body_Lindblad_Hamiltonian(sy, sy) + Jz_1*make_two_body_Lindblad_Hamiltonian(sz, sz)
l2_1 = reshape(l2_1, 4,4,4,4)
l2_2 = Jx_2*make_two_body_Lindblad_Hamiltonian(sx, sx) + Jy_2*make_two_body_Lindblad_Hamiltonian(sy, sy) + Jz_2*make_two_body_Lindblad_Hamiltonian(sz, sz)
l2_2 = reshape(l2_2, 4,4,4,4)

mpo = MPO("-y", params, mpi_cache)
#mpo2 = MPO("+x", params, mpi_cache)
#mpo.A = 2*mpo.A

#Define sampler and optimizer:
sampler = MetropolisSampler(N_MC, burn_in, sweeps, params)
optimizer = TDVPXYZ(sampler, mpo, l1, l2_1, l2_2, τ, ϵ_shift, ϵ_SNR, ϵ_tol, params, ising_int)
NormalizeMPO!(params, optimizer)

#display(optimizer.mpo.A)
#sleep(10000)

# --- plotting additions ---
times_list = Float64[]
mlL2_list = Float64[]
mx_list = Float64[]
my_list = Float64[]
mz_list = Float64[]
Cxx2_list = Float64[]
Cyy2_list = Float64[]
Czz2_list = Float64[]
Cxx3_list = Float64[]
Cyy3_list = Float64[]
Czz3_list = Float64[]
# --- end plotting additions ---

last_iteration_step = 1

#Save parameters to file:
dir = "results/CLRXYZ_uc$(uc_size)_chi$(χ)_N$(N)_J1$(J1)_α1$(α1)_J2$(J2)_α2$(α2)_hx$(hx)_hz$(hz)_γ$(γ)"
if mpi_cache.rank == 0
    if isdir(dir)==true
        cd(dir)
        optimizer = load("optimizer.jld", "optimizer")
        list_of_C = open("C.out", "r")
        last_iteration_step=countlines(list_of_C)+1
    else    
        mkdir(dir)
        cd(dir)
    end
end

MPI.Barrier(mpi_cache.comm)
if mpi_cache.rank != 0
    cd(dir)
end
MPI.bcast(last_iteration_step, 0, mpi_cache.comm)
MPI.bcast(optimizer.τ, 0, mpi_cache.comm)
MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)

if mpi_cache.rank == 0 && last_iteration_step == 1
    
    #Save parameters to parameter file:
    list_of_parameters = open("Ising_decay.params", "w")
    redirect_stdout(list_of_parameters)
    display(params)
    display(sampler)
    display(optimizer)
    close(list_of_parameters)

    mx, my, mz = measure_magnetizations(params, mpo)

    list_of_C = open("C.out", "a")
    println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
    close(list_of_C)

    list_of_obs = open("obs.out", "a")
    println(list_of_obs, mx, ",", my, ",", mz)
    close(list_of_obs)

    Cxx = tensor_correlation(1,2,sx,sx,params,mpo) - mx^2
    Cyy = tensor_correlation(1,2,sy,sy,params,mpo) - my^2
    Czz = tensor_correlation(1,2,sz,sz,params,mpo) - mz^2

    list_of_obs = open("corr2.out", "a")
    println(list_of_obs, Cxx, ",", Cyy, ",", Czz)
    close(list_of_obs)

    Cxx3 = tensor_correlation(1,3,sx,sx,params,mpo) - mx^2
    Cyy3 = tensor_correlation(1,3,sy,sy,params,mpo) - my^2
    Czz3 = tensor_correlation(1,3,sz,sz,params,mpo) - mz^2

    list_of_obs = open("corr3.out", "a")
    println(list_of_obs, Cxx3, ",", Cyy3, ",", Czz3)
    close(list_of_obs)

    M_sq = ( modulated_magnetization_TI(0.0, params, mpo, sz) )#^0.5
    M_stag = ( modulated_magnetization_TI(π, params, mpo, sz))#^0.5
    M_mod = ( modulated_magnetization_TI(2π/params.N, params, mpo, sz))#^0.5

    list_of_obs = open("ssf.out", "a")
    println(list_of_obs, M_sq, ",", M_stag, ",", M_mod)
    close(list_of_obs)

    list_of_times = open("times.out", "a")
    println(list_of_times, 0.0)
    close(list_of_times)

    # --- plotting additions ---
    push!(times_list, 0.0)
    push!(mlL2_list, real(optimizer.optimizer_cache.mlL2)/N)
    push!(mx_list, mx)
    push!(my_list, my)
    push!(mz_list, mz)
    push!(Cxx2_list, Cxx)
    push!(Cyy2_list, Cyy)
    push!(Czz2_list, Czz)
    push!(Cxx3_list, Cxx3)
    push!(Cyy3_list, Cyy3)
    push!(Czz3_list, Czz3)
    # --- end plotting additions ---
end

MPI.Barrier(mpi_cache.comm)
#Random.seed!(mpi_cache.rank)

current_time::Float64 = parse(Float64, readlines("times.out")[end])
while current_time<T

    global last_iteration_step+=1

    Random.seed!(mpi_cache.rank)
    #EulerStep!(optimizer, mpi_cache)
    AdaptiveHeunStepCapped!(max_τ, optimizer, mpi_cache)

    if mpi_cache.rank == 0
    
        #save("optimizer.jld", "optimizer", optimizer)

        mpo = optimizer.mpo

        display(current_time)
        #display(TraceNorm(mpo.A, optimizer))

        list_of_C = open("C.out", "a")
        println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
        close(list_of_C)

        mx, my, mz = measure_magnetizations(params, mpo)
    
        list_of_obs = open("obs.out", "a")
        println(list_of_obs, mx, ",", my, ",", mz)
        close(list_of_obs)
    
        Cxx = tensor_correlation(1,2,sx,sx,params,mpo) - mx^2
        Cyy = tensor_correlation(1,2,sy,sy,params,mpo) - my^2
        Czz = tensor_correlation(1,2,sz,sz,params,mpo) - mz^2
    
        list_of_obs = open("corr2.out", "a")
        println(list_of_obs, Cxx, ",", Cyy, ",", Czz)
        close(list_of_obs)
    
        Cxx3 = tensor_correlation(1,3,sx,sx,params,mpo) - mx^2
        Cyy3 = tensor_correlation(1,3,sy,sy,params,mpo) - my^2
        Czz3 = tensor_correlation(1,3,sz,sz,params,mpo) - mz^2
    
        list_of_obs = open("corr3.out", "a")
        println(list_of_obs, Cxx3, ",", Cyy3, ",", Czz3)
        close(list_of_obs)

        M_sq = ( modulated_magnetization_TI(0.0, 0.0, params, mpo, sz) )#^0.5
        M_stag = ( modulated_magnetization_TI(π, π, params, mpo, sz))#^0.5
        M_mod = ( modulated_magnetization_TI(2π/params.uc_size, 2π/params.uc_size, params, mpo, sz))#^0.5

        list_of_obs = open("ssf.out", "a")
        println(list_of_obs, M_sq, ",", M_stag, ",", M_mod)
        close(list_of_obs)

        global current_time += optimizer.τ
        list_of_times = open("times.out", "a")
        println(list_of_times, current_time)
        close(list_of_times)

        # --- plotting additions ---
        push!(times_list, current_time)
        push!(mlL2_list, real(optimizer.optimizer_cache.mlL2)/N)
        push!(mx_list, mx)
        push!(my_list, my)
        push!(mz_list, mz)
        push!(Cxx2_list, Cxx)
        push!(Cyy2_list, Cyy)
        push!(Czz2_list, Czz)
        push!(Cxx3_list, Cxx3)
        push!(Cyy3_list, Cyy3)
        push!(Czz3_list, Czz3)

        if last_iteration_step % 10 == 0

        p = plot(times_list, mlL2_list, yscale=:log10, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="mlL2/N", xtick=:auto, ytick=:auto)
        savefig(p, "mlL2.png")
        p = plot(times_list, mx_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="mx", xtick=:auto, ytick=:auto)
        savefig(p, "mx.png")
        p = plot(times_list, my_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="my", xtick=:auto, ytick=:auto)
        savefig(p, "my.png")
        p = plot(times_list, mz_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="mz", xtick=:auto, ytick=:auto)
        savefig(p, "mz.png")
        p = plot(times_list, Cxx2_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Cxx_2", xtick=:auto, ytick=:auto)
        savefig(p, "Cxx_2.png")
        p = plot(times_list, Cyy2_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Cyy_2", xtick=:auto, ytick=:auto)
        savefig(p, "Cyy_2.png")
        p = plot(times_list, Czz2_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Czz_2", xtick=:auto, ytick=:auto)
        savefig(p, "Czz_2.png")
        p = plot(times_list, Cxx3_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Cxx_3", xtick=:auto, ytick=:auto)
        savefig(p, "Cxx_3.png")
        p = plot(times_list, Cyy3_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Cyy_3", xtick=:auto, ytick=:auto)
        savefig(p, "Cyy_3.png")
        p = plot(times_list, Czz3_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Czz_3", xtick=:auto, ytick=:auto)
        savefig(p, "Czz_3.png")
        # --- end plotting additions ---

        end

        #save("optimizer_backup.jld", "optimizer", optimizer)
    end

    #sleep(10)

    GC.gc()
end

MPI.Barrier(mpi_cache.comm)

error("Exiting")