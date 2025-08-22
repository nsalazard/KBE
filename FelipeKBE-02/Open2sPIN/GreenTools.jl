"Create GF of a single spin S pointing along vector s0"
function create_spin(S, s0)
    new_s0 = S*normalize!(s0)
    res = sum(K_vec .* new_s0)
    return -im*(res + I*(S+0.5))
end

"""
    create_spinGF(vs, p)

Create initial GF (k and s) for spins of length S pointing along vectors vs[i]. 
Returns two 4x4xN tensors
"""
function create_spinGF(vs, S)
    gk = [create_spin(S, v) for v in vs]
    gk = reduce(hcat, gk)
    gk = reshape(gk, 4,4,:)
    res1 = SymmetricGreenFunction(gk, 1)

    gs = [im*K0 for v in vs]
    gs = reduce(hcat, gs)
    gs = reshape(gs, 4,4,:)
    res2 = AntisymmetricGreenFunction(gs, 1)

    return res1, res2
end

"""
    extract_components(hist)

Produce matrix whose columns are times, x components for all sites, y components
for all sites, z components for all sites, and number of Schwinger bosons for all
sites given a hist::SystemHistory.
"""
function extract_components(hist::SystemHistory)
    res = []
    for t=1:length(hist.times)
        @tullio xcomp[i] := 0.25im*hist.gk[t,t][a,b,i]*Kx[b,a]
        @tullio ycomp[i] := 0.25im*hist.gk[t,t][a,b,i]*Ky[b,a]
        @tullio zcomp[i] := 0.25im*hist.gk[t,t][a,b,i]*Kz[b,a]
        @tullio schbs[i] := 0.5im*hist.gk[t,t][a,a,i]
    
        row = vcat(xcomp, ycomp, zcomp, schbs.-1)
        pushfirst!(row, hist.times[t])
        push!(res, row)
    end
    
    return real(hcat(res...)')
end

"""
    make_complex(arr)

transform a Green function in the real Schwinger boson basis into
the complex basis. Keldysh comes from F, retarded comes from rho/2
"""
function make_spin_complex(arr)
    up_up = arr[1,1] + arr[2,2] + 1im*(arr[2,1] - arr[1,2])
    do_do = arr[3,3] + arr[4,4] + 1im*(arr[4,3] - arr[3,4])
    up_do = arr[1,3] + arr[2,4] + 1im*(arr[2,3] - arr[1,4])
    do_up = arr[3,1] + arr[4,2] + 1im*(arr[4,1] - arr[3,2])

    res = [[up_up, do_up],[up_do, do_do]]
    
    return hcat(res...)
end

"""
    create_history(states, S, sp)
    create_history(states, S, sp, bp)

Create history with initial conditions for all Green functions. If BathParams bp 
are provided, create initial conditions for expectation values related to the 
bosonic environment.
"""
function create_history(states, S, times, sp::SpinParams)
    nothing_here = SymmetricGreenFunction(zeros(Complex, 2,2,2), 1)
    
    gk, gs = create_spinGF(states, S)
    
    Λ = SingleField(compute_Λ(gk[1,1], sp), 2)
    λ = SingleField(zeros(ComplexF64, 3, 1), 2)
    
    Ωk, Ωs = compute_Ω(gk[1,1], gs[1,1])
    Ωk = SymmetricGreenFunction(Ωk, 1)
    Ωs = AntisymmetricGreenFunction(Ωs, 1)
    
    Πk = nothing_here
    Πs = nothing_here
    
    Ms = AntisymmetricGreenFunction(compute_Ms(Ωs[1,1], sp), 0)
    Mk = SymmetricGreenFunction(compute_Mk(Ωk[1,1], sp), 0)
    
    Ds = nothing_here
    Dk = nothing_here
    
    Σk, Σs = compute_Σ(gk[1,1], gs[1,1], Mk[1,1], Ms[1,1], sp)
    Σk = SymmetricGreenFunction(Σk, 1)
    Σs = AntisymmetricGreenFunction(Σs, 1)
    
    hist = SystemHistory(
        gk, gs,
        Mk, Ms,
        Dk, Ds,
        Σk, Σs,
        Ωk, Ωs,
        Πk, Πs,
        Λ, λ,
        [0.0]
    )
    return hist
end

function create_history(states, S, times, sp::SpinParams, bp::BathParams) 
    gk, gs = create_spinGF(states, S)
    
    Λ = SingleField(compute_Λ(gk[1,1], sp), 2)
    λ = SingleField(zeros(ComplexF64, 3, bp.N), 2)
    
    Ωk, Ωs = compute_Ω(gk[1,1], gs[1,1])
    Ωk = SymmetricGreenFunction(Ωk, 1)
    Ωs = AntisymmetricGreenFunction(Ωs, 1)
    
    Πk, Πs = compute_Π(gk[1,1], gs[1,1], bp)
    Πk = SymmetricGreenFunction(Πk, 2)
    Πs = AntisymmetricGreenFunction(Πs, 2)
    
    Ms = AntisymmetricGreenFunction(compute_Ms(Ωs[1,1], sp), 0)
    Mk = SymmetricGreenFunction(compute_Mk(Ωk[1,1], sp), 0)
    
    Ξk, Ξs = compute_Ξ(times, bp)
    bp.Ξk = SingleField(Ξk, 2)
    bp.Ξs = SingleField(Ξs, 2)
    
    Ds = AntisymmetricGreenFunction(2*bp.Ξs[1], 2)
    Dk = SymmetricGreenFunction(2*bp.Ξk[1], 2)
    
    Σk, Σs = compute_Σ(gk[1,1], gs[1,1], Mk[1,1], Ms[1,1], Dk[1,1], Ds[1,1], sp, bp)
    Σk = SymmetricGreenFunction(Σk, 1)
    Σs = AntisymmetricGreenFunction(Σs, 1)
    
    hist = SystemHistory(
        gk, gs,
        Mk, Ms,
        Dk, Ds,
        Σk, Σs,
        Ωk, Ωs,
        Πk, Πs,
        Λ, λ,
        [0.0]
    )
    return hist
end


function initialize_GF(n::Int, nσ::Int)

    N = n*nσ

    gk0 = -1.0im * (I - diagm(ones(N)))
    gs0 = 1.0im * diagm(ones(N))

    gk0 = reshape(gk0, 4,4,:)
    gs0 = reshape(gs0, 4,4,:)
    

    return gk0, gs0
end


function create_history0(times, γ, γc, βL, βR, μL, μR, n, nσ; dt::Real=0.1) 

    gk0, gs0 = initialize_GF(n, nσ)
    # Wrap into GreenFunction containers with first row
    gk = SymmetricGreenFunction(gk0, 1)            # row 1 with gk[1,1]
    gs = AntisymmetricGreenFunction(gs0, 1)   # row 1 with gs[1,1]

    
    hist_Σk = Vector{Vector{NumOrArray}}()
    hist_Σs = Vector{Vector{NumOrArray}}()

    for t1i in 1:lastindex(times)
        new_Σk = Vector{NumOrArray}(undef, t1i+1)
        new_Σs = Vector{NumOrArray}(undef, t1i+1)

        for t2i in 1:t1i
            newsk, newss = compute_Σ0(γ, γc, βL, βR, μL, μR, t1i*dt, t2i*dt, n)
            new_Σk[t2i+1] = newsk
            new_Σs[t2i+1] = newss
        end

        push!(hist_Σk, new_Σk)
        push!(hist_Σs, new_Σs)
    end

    Σk = AntiHermitianGreenFunction(hist_Σk,1)
    Σs = AntiHermitianGreenFunction(hist_Σs,1)

    hist = SystemHistory(
        gk, gs,
        Σk, Σs,
        times
    )
    return hist
end