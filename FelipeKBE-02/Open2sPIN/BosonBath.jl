# Functions related with the surrounding bosonic environment

using Integrals, Cubature

"Type of Bath parameters"
mutable struct BathParams
    "Number of baths"
    N::Int
    "Sites with baths"
    sit::Vector{<:Integer}
    "Ohmic parameters"
    ohm::Vector{<:Real}
    "Bath couplings"
    g::Union{Matrix{<:Real}, Vector{<:Real}}
    "Inverse temperatures"
    KbT::Vector{<:Real}
    "Cutoff frequencies"
    ωc::Vector{<:Real}
    "Bath kernels"
    Ξk::SingleField
    Ξs::SingleField
    "K tensor reduced to only components and sites with bath"
    K::NumOrArray
    
    function BathParams(N::Int, sit::Vector{<:Integer}, ohm::Vector{<:Real}, g::Union{Matrix{<:Real}, Vector{<:Real}}, KbT::Vector{<:Real}, ωc::Vector{<:Real})
        x = SingleField(zeros(1), 2)
        y = SingleField(zeros(1), 2)
        
        ind = zeros(3, N)
        for (j,i) in enumerate(ind)
            ind[j] = (g[j] == 0) ? 0.0 : 1.0
        end
        
        @tullio K_red[a,b,α,i] := ind[α,i]*K_tens[a,b,α]
        
        return new(N, sit, ohm, g, KbT, ωc, x, y, K_red)
    end
    
end

"s-Ohmic spectral function"
function ohmic_j(ω,γ,s,ωc)
    return γ*ωc^(1-s)*ω^s*exp(-abs(ω)/ωc)
end

"Integrand for computing Keldysh bath kernel"
function integrand_Ξk(out, ω, p)
    bp, t = p
    for i=1:bp.N
        if bp.KbT[i] == 0
            for α=1:3
                # Missing multiplication by 1im due to integrator details
                out[α,i] = -ohmic_j(ω,bp.g[α,i],bp.ohm[i],bp.ωc[i])*cos(ω*t)/(2π)
            end
        else
            for α=1:3
                # Missing multiplication by 1im due to integrator details
                out[α,i] = -coth(0.5*ω/bp.KbT[i]) * ohmic_j(ω,bp.g[α,i],bp.ohm[i],bp.ωc[i]) * cos(ω*t)/(2π)
            end
        end
    end
end

function integrand_Ξs(out, ω, p)
    bp, t = p
    for i=1:bp.N
        for α=1:3
            # Missing multiplication by 1im due to integrator details
            out[α,i] = -ohmic_j(ω,bp.g[α,i],bp.ohm[i],bp.ωc[i]) * sin(ω*t)/π
        end
    end
end

"""
    compute_Ξ(times, bp)

Compute Keldysh and spectral bath kernel. Keldysh is integrated using Cubature and spectral is given from analytic solution for s-Ohmic spectral function. Returns vector of tensors Ξ[α,i].
"""
function compute_Ξ(times, bp::BathParams)
    prototype = zeros(3, bp.N)
    domain = (0, Inf)
    func1 = IntegralFunction(integrand_Ξk, prototype)
    func2 = IntegralFunction(integrand_Ξs, prototype)
    
    Ξk = Vector{NumOrArray}()
    Ξs = Vector{NumOrArray}()
    
    ran = collect(times)
    pushfirst!(ran, 0.0)
    
    for t in ran
        prob = IntegralProblem(func1, domain, (bp, t))
        sol = solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3)
        push!(Ξk, 1im*sol.u)
        
        prob = IntegralProblem(func2, domain, (bp, t))
        sol = solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3)
        push!(Ξs, sol.u)
    end
    
    return Ξk, Ξs
end 

function compute_λ(hist::SystemHistory, bp::BathParams, t1, h)
    new_λ = zeros(ComplexF64, 3, bp.N)
    
    func1 = bp.Ξs.ev[1:t1]
    func2 = time_diag(hist.gk)[1:t1]
    
    list1 = [Array{ComplexF64}(undef, size(new_λ)) for i=1:length(func1)]
    list2 = [Array{ComplexF64}(undef, size(new_λ)) for i=1:length(func1)]
    
    for k=1:length(list1)
        x = list1[k]
        f = func2[k]
        @tullio x[α,i] = 0.5*1im*K_tens[a,b,α]*view(f, :,:,bp.sit)[a,b,i]
    end
    
    for k=1:length(list2)
        x = list2[k]
        f1 = func1[t1-k+1]
        f2 = list1[k]
        @tullio x[α,i] = f1[α,i]*f2[α,i]
    end
    
    new_λ .+= trapz(list2, 1, t1, h)
    
    return new_λ
end
    
"Compute Π self energies."
function compute_Π(gk, gs, bp)
    @tullio ok1[a,c,α,i] := bp.K[a,b,α,i]*view(gk, :,:,bp.sit)[b,c,i]
    @tullio ok2[c,a,α,i] := bp.K[c,d,α,i]*view(gk, :,:,bp.sit)[a,d,i]
    @tullio ok[α,i] := ok1[a,c,α,i]*ok2[c,a,α,i]
    
    @tullio os1[a,c,α,i] := bp.K[a,b,α,i]*view(gs, :,:,bp.sit)[b,c,i]
    @tullio os2[c,a,α,i] := bp.K[c,d,α,i]*view(gs, :,:,bp.sit)[a,d,i]
    @tullio ok[α,i] += 0.25*os1[a,c,α,i]*os2[c,a,α,i]
    
    @tullio os[α,i] := ok1[a,c,α,i]*os2[c,a,α,i]
    
    return 0.125im*ok, 0.25im*os
end

"""
    compute_dρ(hist, bp, t1,t2,h, aux1,aux2)
    compute_dρ(bp)

Compute Dρ(t1, t2) Green functions of the Hubbard-Stratonovich field for the spin-boson interaction. 
Integrals are done with trapezoid method with timestep h. aux1 and aux2 are used to hold temporary
variables. If only bp is passed, then integrals are ignored.
"""
function compute_Ds(hist::SystemHistory, bp::BathParams, t1,t2,h)
    new_D = zeros(ComplexF64, 3, bp.N)
    new_D += 2*bp.Ξs[t1-t2+1]
    
    if t1==t2 return new_D end
    
    func1 = bp.Ξs.ev[1:t1]
    func3 = hist.Ds[:,t2]
      
    list1 = [zero(new_D) for i=1:length(func1)]
    list2 = [zero(new_D) for i=1:length(func1)]
        
    for tn in t2:t1-1
        func2 = hist.Πs[tn,:]
        
        for k in t2:tn
            x = list1[k]
            f1 = func2[k]
            f2 = func3[k]            
            x .= f1 .* f2
        end
                
        list2[tn] = trapz(list1, t2, tn, h)
    end
    
    func2 = hist.Πs[t1,:]
    
    for k in t2:t1-1
        x = list1[k]
        f1 = func2[k]
        f2 = func3[k]
        x .= f1 .* f2
    end
        
    list2[t1] = trapz(list1, t2, t1, h, final_step=false)
            
    for k in t2:t1
        x = list1[k]
        f1 = func1[t1-k+1]
        f2 = list2[k]
        x .= f1 .* f2
    end
    
    new_D += 2*trapz(list1, t2, t1, h)
    
    inv_factor = 0.5*h^2*bp.Ξs[1] .* hist.Πs[t1,t1]
    inv_factor = 1 ./ (1 .- inv_factor)
        
    res = inv_factor .* new_D
    
    return res
end

function compute_Dk(hist::SystemHistory, bp::BathParams, t1,t2,h)
    new_D = zeros(ComplexF64, 3, bp.N)
    new_D = 2*bp.Ξk[t1-t2+1]
    
    if t1==1 return new_D end
    
    func1 = bp.Ξs.ev[1:t1]
    func3 = hist.Dk[:,t2]
      
    list1 = [zero(new_D) for i=1:length(func1)]
    list2 = [zero(new_D) for i=1:length(func1)]
    
    for tn in 1:t1-1
        func2 = hist.Πs[tn,:]
        
        for k in 1:tn
            x = list1[k]
            f1 = func2[k]
            f2 = func3[k]            
            x .= f1 .* f2
        end
        
        list2[tn] = trapz(list1, 1, tn, h)
    end
    
    func2 = hist.Πs[t1,:]
    
    for k in 1:t1-1
        x = list1[k]
        f1 = func2[k]
        f2 = func3[k]
        x .= f1 .* f2
    end
        
    list2[t1] = trapz(list1, 1, t1, h, final_step=false)
            
    for k in 1:t1
        x = list1[k]
        f1 = func1[t1-k+1]
        f2 = list2[k]
        x .= f1 .* f2
    end
    
    new_D += 2*trapz(list1, 1, t1, h)
    
    if t2 != 1
    
        func1 = bp.Ξk.ev[1:t1]
        func3 = hist.Ds[:,t2]

        list1 = [zero(new_D) for i=1:length(func1)]
        list2 = [zero(new_D) for i=1:length(func1)]

        for tn in 1:t2
            func2 = hist.Πs[:,tn]

            for k in 1:tn
                x = list1[k]
                f1 = func1[t1-k+1]
                f2 = func2[k]            
                x .= f1 .* f2
            end

            list2[tn] = trapz(list1, 1, tn, h)
        end

        for k in 1:t2
            x = list1[k]
            f1 = list2[k]
            f2 = func3[k]
            x .= f1 .* f2
        end

        new_D += 2*trapz(list1, 1, t2, h)

        func1 = bp.Ξs.ev[1:t1]
        func3 = hist.Ds[:,t2]

        list1 = [zero(new_D) for i=1:length(func1)]
        list2 = [zero(new_D) for i=1:length(func1)]

        for tn in 1:t1
            func2 = hist.Πk[tn,:]

            for k in 1:t2
                x = list1[k]
                f1 = func2[k]
                f2 = func3[k]            
                x .= f1 .* f2
            end

            list2[tn] = trapz(list1, 1, t2, h)
        end

        for k in 1:t1
            x = list1[k]
            f1 = func1[t1-k+1]
            f2 = list2[k]
            x .= f1 .* f2
        end

        new_D += -2*trapz(list1, 1, t1, h)
    end
    
    inv_factor = 0.5*h^2*bp.Ξs[1] .* hist.Πs[t1,t1]
    inv_factor = 1 ./ (1 .- inv_factor)
        
    res = inv_factor .* new_D
    
    return res
end

"Compute Σ self energies."
function compute_Σ(gk, gs, Mk, Ms, Dk, Ds, sp::SpinParams, bp::BathParams)
    # Contractions are dramatically faster and less memory-intensive when done one
    # pair of tensors at a time.
        
    @tullio aux1[a,c,α,i] := K_tens[a,b,α]*gk[b,c,i]
    @tullio sk1[a,d,α,β,i] := aux1[a,c,α,i]*K_tens[c,d,β]
    
    @tullio aux2[α,γ,i,k] := sp.J[α,β,i,j]*Mk[β,γ,j,k]
    @tullio sk2[α,δ,i] := aux2[α,γ,i,k]*sp.J[γ,δ,k,i]
    
    @tullio aux1[a,c,α,i] = K_tens[a,b,α]*gs[b,c,i]
    @tullio sk3[a,d,α,β,i] := aux1[a,c,α,i]*K_tens[c,d,β]
    
    @tullio aux2[α,γ,i,k] = sp.J[α,β,i,j]*Ms[β,γ,j,k]
    @tullio sk4[α,δ,i] := aux2[α,γ,i,k]*sp.J[γ,δ,k,i]

    @tullio sk[a,b,i] := sk1[a,b,α,β,i]*sk2[α,β,i] + 0.25*sk3[a,b,α,β,i]*sk4[α,β,i]
    @tullio ss[a,b,i] := sk1[a,b,α,β,i]*sk4[α,β,i] + sk3[a,b,α,β,i]*sk2[α,β,i]
    
    vsk = view(sk, :,:,bp.sit)
    vss = view(ss, :,:,bp.sit)
    
    @tullio vsk[a,b,i] += view(sk1, :,:,:,:,bp.sit)[a,b,α,α,i]*Dk[α,i]
    @tullio vsk[a,b,i] += 0.25*view(sk3, :,:,:,:,bp.sit)[a,b,α,α,i]*Ds[α,i]
    
    @tullio vss[a,b,i] += view(sk1, :,:,:,:,bp.sit)[a,b,α,α,i]*Ds[α,i]
    @tullio vss[a,b,i] += view(sk3, :,:,:,:,bp.sit)[a,b,α,α,i]*Dk[α,i]
    
    return 0.25im*sk, 0.25im*ss
end

"In-place computation of the Hamiltonian tensor at time t"
function compute_hamiltonian!(out, hist::SystemHistory, sp::SpinParams, bp::BathParams, t)
    # Field-term comes first because it redefines all elements
    @tullio out[a,b,i] = 0.25*hist.Λ[t][α,i]*K_tens[a,b,α]
    @tullio out[a,b,i] += 0.25*sp.B[α]*K_tens[a,b,α]
    
    sliced_ham = view(out, :,:,bp.sit)    
    @tullio sliced_ham[a,b,i] += 0.25*hist.λ[t][α,i]*K_tens[a,b,α]
end






