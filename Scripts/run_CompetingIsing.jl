include("../VMPOMC.jl")
using .VMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI
using Dates
using JLD


mpi_cache = set_mpi()

#Set parameters:
Jx= 0.0 #interaction strength
Jy= 0.0 #interaction strength
Jz= 0.0 #interaction strength
J1= -1.0 #interaction strength
J2= -0.5 #interaction strength
hx= 0.5 #transverse field strength
hz= 0.0 #longitudinal field strength
γ = 1.0 #spin decay rate
N = 12 #number of spins
α1 = 1
α2 = 3
γ_d = 0

#Set hyperparameters:
χ = 6 #MPO bond dimension
burn_in = 2 #Monte Carlo burn-in
δ = 0.02 #step size
ϵ = 0.01
N_iterations = 500
uc_size = 1
N_MC = 2500 #number of Monte Carlo samples
ising_int = "CompetingIsing"

params = Parameters(N,χ,Jx,Jy,Jz,J1,J2,hx,hz,γ,γ_d,α1,α2,uc_size)


#Define one-body Lindbladian operator:
const l1 = make_one_body_Lindbladian(hx*sx+hz*sz, sqrt(γ)*smx)

#Save parameters to file:
dir = "results/Competing_Ising_decay_uc$(uc_size)_chi$(χ)_N$(N)_J1$(J1)_α1$(α1)_J2$(J2)_α2$(α2)_hx$(hx)_hz$(hz)_γ$(γ)"

if mpi_cache.rank == 0
    if isdir(dir)==true
        error("Directory already exists")
    end
    mkdir(dir)
    cd(dir)
end

Random.seed!(mpi_cache.rank)
A_init = zeros(ComplexF64, uc_size,χ,χ,4)
for n in 1:uc_size
    A_init[n,:,:,1].=1.0
    A_init[n,:,:,2].=1.0
    A_init[n,:,:,3].=1.0
    A_init[n,:,:,4].=1.0
end
#A_init+= 0.01*rand(ComplexF64, uc_size,χ,χ,4)
mpo = MPO(A_init)

#Define sampler and optimizer:
sampler = MetropolisSampler(N_MC, burn_in)
optimizer = TDVP(sampler, mpo, l1, ϵ, params, ising_int)
normalize_MPO!(params, optimizer)

if mpi_cache.rank == 0
    #Save parameters to parameter file:
    list_of_parameters = open("Ising_decay.params", "w")
    redirect_stdout(list_of_parameters)
    display(params)
    display(sampler)
    #display(optimizer)
    println("\nN_iter\t", N_iterations)
    println("δ\t\t", δ)
    close(list_of_parameters)
end

mlL2_list = []
mx_list = []
my_list = []
mz_list = []
S2_list = []
Cxx_list = []
Czz_list = []
C2x_list = []
C2z_list = []
CX_list = []
Mx_stag_list = []
Mx_sq_list = []
Mx_mod_list = []
Mz_stag_list = []
Mz_sq_list = []
Mz_mod_list = []


for k in 1:N_iterations

    #Optimize MPO:
    tensor_compute_gradient!(optimizer)
    MPI_mean!(optimizer, mpi_cache)
    if mpi_cache.rank == 0
        optimize!(optimizer, δ)
    end
    MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)

    if mpi_cache.rank == 0
        #Calculate steady-state magnetizations:
#        mx = real.( tensor_magnetization(2,params,mpo,sx) )
#        my = real.( tensor_magnetization(2,params,mpo,sy) )
#        mz = real.( tensor_magnetization(2,params,mpo,sz) )

        Z = tensor_purity(params,mpo)

        S2 = real( -log(Z)/N )

        mx=0.0
        my=0.0
        mz=0.0
        for n in 1:1#1:uc_size
            mx += real.( tensor_magnetization(n,params,mpo,sx) )
            my += real.( tensor_magnetization(n,params,mpo,sy) )
            mz += real.( tensor_magnetization(n,params,mpo,sz) )
        end
        mx/=uc_size
        my/=uc_size
        mz/=uc_size

        #Cxx = tensor_cummulant(1, 2, sx, sx, params, mpo)
        #Czz = tensor_cummulant(1, 2, sz, sz, params, mpo)

        Cxx = [tensor_cummulant(1, j, sx, sx, params, mpo) for j=2:params.N÷2+1]
        Czz = [tensor_cummulant(1, j, sz, sz, params, mpo) for j=2:params.N÷2+1]

        #C2x = C2(sx,sx,params,mpo)
        #C2z = C2(sz,sz,params,mpo)

        Mx_sq = ( squared_magnetization(params, mpo, sx) )^0.5
        Mx_stag = ( squared_staggered_magnetization(params, mpo, sx) )^0.5
        Mx_mod = ( modulated_magnetization(2π/params.N, params, mpo, sx) )^0.5
        Mz_sq = ( squared_magnetization(params, mpo, sz) )^0.5
        Mz_stag = ( squared_staggered_magnetization(params, mpo, sz) )^0.5
        Mz_mod = ( modulated_magnetization(2π/params.N, params, mpo, sz) )^0.5
        #M_stag_e = efficient_squared_staggered_magnetization(params, mpo, sz)

        list_of_C = open("C.out", "a")
        println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
        close(list_of_C)

        list_of_obs = open("obs.out", "a")
        println(list_of_obs, mx, ",", my, ",", mz, ",", Z)
        close(list_of_obs)


        push!(mlL2_list, real(optimizer.optimizer_cache.mlL2)/N)
        push!(mx_list, mx)
        push!(my_list, my)
        push!(mz_list, mz)
        push!(S2_list, S2)
        push!(Mx_sq_list, Mx_sq)
        push!(Mx_stag_list, Mx_stag)
        push!(Mx_mod_list, Mx_mod)
        push!(Mz_sq_list, Mz_sq)
        push!(Mz_stag_list, Mz_stag)
        push!(Mz_mod_list, Mz_mod)
        #push!(M_stag_e_list, M_stag_e)
        #push!(Cxx_list, Cxx)
        #push!(Czz_list, Czz)
        push!(Cxx_list, Cxx)
        push!(Czz_list, Czz)
        #push!(C2x_list, C2x)
        #push!(C2z_list, C2z)

        if mod(k,10)==0
            p = plot(mlL2_list, yscale=:log10)
            savefig(p, "mlL2.png")
            p = plot(my_list)
            savefig(p, "my.png")
            p = plot(mx_list)
            savefig(p, "mx.png")
            p = plot(mz_list)
            savefig(p, "mz.png")
            p = plot(S2_list)
            savefig(p, "S2.png")
            p = plot(Mx_sq_list)
            savefig(p, "M_sq.png")
            p = plot(Mx_stag_list)
            savefig(p, "M_stag.png")
            p = plot(Mx_mod_list)
            savefig(p, "M_mod.png")
            p = plot(Mz_sq_list)
            savefig(p, "M_sq.png")
            p = plot(Mz_stag_list)
            savefig(p, "M_stag.png")
            p = plot(Mz_mod_list)
            savefig(p, "M_mod.png")
            #p = plot(Cxx_list)
            #savefig(p, "Cxx.png")
            #p = plot(Czz_list)
            #savefig(p, "Czz.png")
            #p = plot(C2x_list)
            #savefig(p, "C2x.png")
            #p = plot(C2z_list)
            #savefig(p, "C2z.png")
            for j=1:params.N÷2
                p = plot(map(v -> v[j], Cxx_list))
                savefig(p, "Cxx_$(j).png")
                p = plot(map(v -> v[j], Czz_list))
                savefig(p, "Czz_$(j).png")
            end
        end
    end

    if mod(k,10)==0
        GC.gc()
    end
end

exit()