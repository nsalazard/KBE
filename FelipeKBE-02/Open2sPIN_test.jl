###############
# 1. Imports
###############
include("./Open2sPIN/Open2PIN.jl")
using .Open2PIN

using Tullio
using DelimitedFiles
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using QuadGK


###############
# 2. Parameters
###############
ω        = 2.0 * pi / 10.0   # frequency of the trajectory
dt       = 0.1
t_start  = 0.1
Nt       = 120
tilt_deg = 15
n        = 2
nσ       = 2
γ, γc    = 1.0, 1.0
βL, βR   = 1/0.026, 1/0.026
μL, μR   = 0.0, 0.0
Jsc      = 0.1
t_hop    = 1.0
ti       = 40

file_spin = "density_spin_120.txt"
file_occ  = "occupation_vs_time_120.txt"

###############
# 3. Trajectory & History
###############
traj  = sample_trajectory(ω=ω, dt=dt, a=t_start, b=Nt, tilt_deg=tilt_deg)
times = collect(t_start:dt:Nt)
println("Times: ", size(times))

hist = create_history0(times, γ, γc, βL, βR, μL, μR, n, nσ, t_start, Nt, ti; dt=dt)

@time evolve_pc0!(hist, times, n, traj, Jsc, t_hop)


###############
# 4. Helper Functions
###############

# Build density matrix from gk
function density_matrix_from_gk(gk, t_index)
    G = gk[t_index, t_index]                 # 4D array: (n, σ, n', σ')
    dims = size(G)                           # (Nn, Nσ, Nn', Nσ')
    @assert length(dims) == 4
    Nn, Nn2, Ns, Ns2 = dims
    @assert Nn == Nn2 && Ns == Ns2           # usually same dims for primed indices

    rho = zeros(ComplexF64, Ns, Ns, Nn, Nn)  # rho[σ,σ',n,n']
    for s in 1:Ns, s2 in 1:Ns, n in 1:Nn, n2 in 1:Nn
        delta = (n == n2 && s == s2) ? 1.0 : 0.0
        rho[s, s2, n, n2] = 0.5 * (delta - 1im * G[s, s2, n, n2])
    end
    return rho
end

# On-site occupation (sum over spin)
function onsite_occupations(rho)
    Ns, _, Nn, _ = size(rho)
    occ = zeros(ComplexF64, Nn)
    for n in 1:Nn, s in 1:Ns
        occ[n] += rho[s, s, n, n]
    end
    return occ
end

# Pauli matrices
const pauli_x = [0 1; 1 0]
const pauli_y = [0 1im; -1im 0]
const pauli_z = [1 0; 0 -1]
const pauli   = [pauli_x, pauli_y, pauli_z]

# Spin density
function spin_density(rho)
    Ns, _, Nn, _ = size(rho)
    spin = zeros(ComplexF64, 3, Nn)   # spin[component, site]
    for n in 1:Nn
        ρn = rho[:, :, n, n]
        for ii in 1:3
            spin[ii, n] = 0.5 * tr(pauli[ii] * ρn)
        end
    end
    return spin
end


###############
# 5. Time Evolution: Spin vs Time
###############
spin_vs_time = [Float64[] for _ in 1:3]  # (Sx, Sy, Sz)

for t in 1:length(times)
    rho  = density_matrix_from_gk(hist.gk, t)
    spin = spin_density(rho)
    push!(spin_vs_time[1], real(spin[1,1]))   # Sx (site 1)
    push!(spin_vs_time[2], real(spin[2,1]))   # Sy (site 1)
    push!(spin_vs_time[3], real(spin[3,1]))   # Sz (site 1)
end

Sx, Sy, Sz = spin_vs_time


###############
# 6. Save to file
###############
data = hcat(times, Sx, Sy, Sz)   # matrix with columns [t Sx Sy Sz]
writedlm(file_spin, data, ' ')
println("Saved results to ", file_spin)

###############
# 7. On-site occupation vs. time
###############
occ_vs_time = Float64[]   # occupation at site 1 vs time

for t in 1:length(times)
    rho = density_matrix_from_gk(hist.gk, t)
    occ = onsite_occupations(rho)
    push!(occ_vs_time, real(occ[1]))   # occupation at site 1
end

# Save to file
occ_data = hcat(times, occ_vs_time)   # columns: [t occ]
writedlm(file_occ, occ_data, ' ')
println("Saved results to ", file_occ)
###############