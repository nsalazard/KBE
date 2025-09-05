"""
    evolve_pc!(hist::SystemHistory, new_times, sp::SpinParams)
    evolve_pc!(hist::SystemHistory, new_times, sp::SpinParams, bp::BathParams)
    
Evolve hist for new_times using a predictor-corrector algorithm. If BathParams bp are
provided, evolution includes effect of bosonic environment.
"""
# Only spins

function evolve_pc0!(hist::SystemHistory, new_times, n::Int, traj::AbstractVector{<:SVector{3,Float64}}, Jsc::Real, t_hop::Real)
    
    tstep = new_times[1]
    
    # This array will be reused to hold the Hamiltonian tensor
    ham = zeros(ComplexF64, 2, 2, n, n)

    for t1 = 1:lastindex(new_times)
        println("Current time-step for t is $t1")
        if t1%10==0 println("Current time-step for t is $t1"); flush(stdout) end
        
        # Compute Hamiltonian
        compute_hamiltonian!(ham; traj = traj[t1], Jsc=Jsc, t_hop=t_hop, n=n)
        
        # ---------------------------------------------------------------------------
        # Predict gk and gs
        
        ∂gk_v = Vector{NumOrArray}(undef, t1)
        ∂gs_v = Vector{NumOrArray}(undef, t1)

        for t2 = 1:t1
            dkv, dsv = rhs_vertical(hist, ham, t1, t2, tstep)
            ∂gk_v[t2] = dkv
            ∂gs_v[t2] = dsv

            ####################################
            if any(isnan, ∂gk_v[t2])
                error("NaN detected in ∂gk_v at t1=$t1, t2=$t2")
            end
            if any(isnan, ∂gs_v[t2])
                error("NaN detected in ∂gs_v at t1=$t1, t2=$t2")
            end
            ##################################
        end

        #∂gk_d = rhs_diag(∂gk_v[t1])
        ∂gk_d = rhs_diag(hist, ham, t1, tstep)

        new_gk = Vector{NumOrArray}(undef, t1+1)
        new_gs = Vector{NumOrArray}(undef, t1+1)

        for t2 = 1:t1
            new_gk[t2] = hist.gk[t1,t2] + tstep*∂gk_v[t2]
            new_gs[t2] = hist.gs[t1,t2] + tstep*∂gs_v[t2]

            ################################
            if any(isnan, new_gk[t2])
                error("NaN detected in new_gk at t1=$t1, t2=$t2")
            end
            if any(isnan, new_gs[t2])
                error("NaN detected in new_gs at t1=$t1, t2=$t2")
            end
            #################################
        end

        new_gk[t1+1] = hist.gk[t1,t1] + tstep*∂gk_d
        new_gs[t1+1] = hist.gs[1,1]

        push!(hist.gk, new_gk)
        push!(hist.gs, new_gs)
        
        # ---------------------------------------------------------------------------
        # Start of the correction phase
        
        for rep = 1:1
            compute_hamiltonian!(ham; traj = traj[t1+1], Jsc=Jsc, t_hop=t_hop, n=n)

            ∂gk_v1 = Vector{NumOrArray}(undef, t1)
            ∂gs_v1 = Vector{NumOrArray}(undef, t1)
            for t2 = 1:t1
                dkv1, dsv1 = rhs_vertical(hist, ham, t1+1, t2, tstep)
                ∂gk_v1[t2] = dkv1
                ∂gs_v1[t2] = dsv1

                ####################################
                if any(isnan, ∂gk_v1[t2])
                error("NaN detected in ∂gk_v1 at t1=$t1, t2=$t2")
                end
                if any(isnan, ∂gs_v1[t2])
                    error("NaN detected in ∂gs_v1 at t1=$t1, t2=$t2")
                end
            ##########################################
            end

            

            #∂gk_d1 = rhs_diag(∂gk_v1[t1])
            ∂gk_d1 = rhs_diag(hist, ham, t1, tstep)

            for t2=1:t1
                hist.gk[t1+1,t2] = hist.gk[t1,t2] + 0.5*tstep*(∂gk_v[t2] + ∂gk_v1[t2])
                hist.gk[t1+1,t2] = hist.gs[t1,t2] + 0.5*tstep*(∂gs_v[t2] + ∂gs_v1[t2]);

                #################################
                if any(isnan, hist.gk[t1+1,t2])
                error("NaN detected in gk at t1=$t1, t2=$t2")
                end
                if any(isnan, hist.gs[t1+1,t2])
                    error("NaN detected in gs at t1=$t1, t2=$t2")
                end
                ################################

            end


            hist.gk[t1+1,t1+1] = hist.gk[t1,t1] + 0.5*tstep*(∂gk_d + ∂gk_d1)

        end
    end
    #append!(hist.times, new_times)
end



function evolve_pc!(hist::SystemHistory, new_times, sp::SpinParams)
    tstep = new_times[1]
    
    # This array will be reused to hold the Hamiltonian tensor
    ham = Array{ComplexF64}(undef, 4,4,sp.N)
    
    for t1 = 1:lastindex(new_times)
        if t1%10==0 println("Current time-step for t is $t1"); flush(stdout) end
        
        # Compute Hamiltonian
        
        compute_hamiltonian!(ham, hist, sp, t1)
        
        # ---------------------------------------------------------------------------
        # Predict gk and gs
        
        ∂gk_v = Vector{NumOrArray}(undef, t1)
        ∂gs_v = Vector{NumOrArray}(undef, t1)

        for t2 = 1:t1
            dkv, dsv = rhs_vertical(hist, ham, t1, t2, tstep)
            ∂gk_v[t2] = dkv
            ∂gs_v[t2] = dsv
        end
        
        ∂gk_d = rhs_diag(hist, ham, t1, tstep)
        
        new_gk = Vector{NumOrArray}(undef, t1+1)
        new_gs = Vector{NumOrArray}(undef, t1+1)
        
        for t2 = 1:t1
            new_gk[t2] = hist.gk[t1,t2] + tstep*∂gk_v[t2]
            new_gs[t2] = hist.gs[t1,t2] + tstep*∂gs_v[t2]
        end

        new_gk[t1+1] = hist.gk[t1,t1] + tstep*∂gk_d
        new_gs[t1+1] = hist.gs[1,1]
                
        push!(hist.gk, new_gk)
        push!(hist.gs, new_gs)
        
        # ---------------------------------------------------------------------------
        # Compute the mean field
        
        new_Λ = compute_Λ(hist.gk[t1+1,t1+1], sp)

        push!(hist.Λ, new_Λ)
        
        # ---------------------------------------------------------------------------
        # Compute mean-field self-energies
        
        new_Ωk = Vector{NumOrArray}(undef, t1+1)
        new_Ωs = Vector{NumOrArray}(undef, t1+1)

        for t2 = 1:t1+1
            newok, newos = compute_Ω(hist.gk[t1+1,t2], hist.gs[t1+1,t2])
            new_Ωk[t2] = newok
            new_Ωs[t2] = newos
        end
        
        push!(hist.Ωk, new_Ωk)
        push!(hist.Ωs, new_Ωs)
        
        # ---------------------------------------------------------------------------
        # Compute mean-field propagators

        new_Mk = Vector{NumOrArray}(undef, t1+1)
        new_Ms = Vector{NumOrArray}(undef, t1+1)
        
        # Ms needs to be computed first, as it only depends on itself.
        
        for t2 = 1:t1+1
            new_Ms[t2] = compute_Ms(hist, sp, t1+1, t2, tstep)
        end
        
        push!(hist.Ms, new_Ms)
        
        # Mk needs to be computed second as it depends on Ms. However, Mk[t1+1,t1+1]
        # needs to be computed separately after the loop because the integral 
        # required calls all the Mk[t1+1,1:t1].
        
        for t2 = 1:t1
            new_Mk[t2] = compute_Mk(hist, sp, t1+1, t2, tstep)
        end
        
        push!(hist.Mk, new_Mk)
        
        hist.Mk[t1+1,t1+1] = zeros(ComplexF64, 3,3,sp.N,sp.N)
        hist.Mk[t1+1,t1+1] = compute_Mk(hist, sp, t1+1, t1+1, tstep)
        
        # ---------------------------------------------------------------------------
        # Compute spin self-energies
        
        new_Σk = Vector{NumOrArray}(undef, t1+1)
        new_Σs = Vector{NumOrArray}(undef, t1+1)
        
        for t2 = 1:t1+1
            newsk, newss = compute_Σ(hist.gk[t1+1,t2],hist.gs[t1+1,t2],hist.Mk[t1+1,t2],hist.Ms[t1+1,t2],sp)
            new_Σk[t2] = newsk
            new_Σs[t2] = newss
        end
        
        push!(hist.Σk, new_Σk)
        push!(hist.Σs, new_Σs)
        
        # ---------------------------------------------------------------------------
        # Start of the correction phase
        
        for rep = 1:1
            compute_hamiltonian!(ham, hist, sp, t1+1)

            ∂gk_v1 = Vector{NumOrArray}(undef, t1)
            ∂gs_v1 = Vector{NumOrArray}(undef, t1)
            for t2 = 1:t1
                dkv1, dsv1 = rhs_vertical(hist, ham, t1+1, t2, tstep)
                ∂gk_v1[t2] = dkv1
                ∂gs_v1[t2] = dsv1
            end

            ∂gk_d1 = rhs_diag(hist, ham, t1+1, tstep)

            for t2=1:t1
                hist.gk[t1+1,t2] = hist.gk[t1,t2] + 0.5*tstep*(∂gk_v[t2] + ∂gk_v1[t2])
                hist.gs[t1+1,t2] = hist.gs[t1,t2] + 0.5*tstep*(∂gs_v[t2] + ∂gs_v1[t2]);
            end

            hist.gk[t1+1,t1+1] = hist.gk[t1,t1] + 0.5*tstep*(∂gk_d + ∂gk_d1)

            hist.Λ[t1+1] = compute_Λ(hist.gk[t1+1,t1+1], sp)

            for t2 = 1:t1+1
                newok, newos = compute_Ω(hist.gk[t1+1,t2], hist.gs[t1+1,t2])
                hist.Ωk[t1+1,t2] = newok
                hist.Ωs[t1+1,t2] = newos
            end
            
            for t2 = 1:t1+1
                hist.Ms[t1+1,t2] = compute_Ms(hist, sp, t1+1, t2, tstep)
            end

            for t2 = 1:t1+1
                hist.Mk[t1+1,t2] = compute_Mk(hist, sp, t1+1, t2, tstep)
            end
            
            for t2 = 1:t1+1
                newsk, newss = compute_Σ(hist.gk[t1+1,t2],hist.gs[t1+1,t2],hist.Mk[t1+1,t2],hist.Ms[t1+1,t2],sp)
                hist.Σk[t1+1,t2] = newsk
                hist.Σs[t1+1,t2] = newss
            end
        end
    end
    append!(hist.times, new_times)
end


# Spins + bath
function evolve_pc!(hist::SystemHistory, new_times, sp::SpinParams, bp::BathParams)
    tstep = new_times[1]
    
    # This array will be reused to hold the Hamiltonian tensor
    ham = Array{ComplexF64}(undef, 4,4,sp.N)
    
    for t1 = 1:lastindex(new_times)
        if t1%10==0 println("Current time-step for t is $t1"); flush(stdout) end
        
        # Compute Hamiltonian
        
        compute_hamiltonian!(ham, hist, sp, bp, t1)
        
        # ---------------------------------------------------------------------------
        # Predict gk and gs
        
        ∂gk_v = Vector{NumOrArray}(undef, t1)
        ∂gs_v = Vector{NumOrArray}(undef, t1)

        for t2 = 1:t1
            dkv, dsv = rhs_vertical(hist, ham, t1, t2, tstep)
            ∂gk_v[t2] = dkv
            ∂gs_v[t2] = dsv
        end
        
        ∂gk_d = rhs_diag(hist, ham, t1, tstep)
        
        new_gk = Vector{NumOrArray}(undef, t1+1)
        new_gs = Vector{NumOrArray}(undef, t1+1)
        
        for t2 = 1:t1
            new_gk[t2] = hist.gk[t1,t2] + tstep*∂gk_v[t2]
            new_gs[t2] = hist.gs[t1,t2] + tstep*∂gs_v[t2]
        end

        new_gk[t1+1] = hist.gk[t1,t1] + tstep*∂gk_d
        new_gs[t1+1] = hist.gs[1,1]
                
        push!(hist.gk, new_gk)
        push!(hist.gs, new_gs)
        
        # ---------------------------------------------------------------------------
        # Compute the mean field
        
        new_Λ = compute_Λ(hist.gk[t1+1,t1+1], sp)
        new_λ = compute_λ(hist, bp, t1, tstep)

        push!(hist.Λ, new_Λ)
        push!(hist.λ, new_λ)

        
        # ---------------------------------------------------------------------------
        # Compute mean-field self-energies
        
        new_Ωk = Vector{NumOrArray}(undef, t1+1)
        new_Ωs = Vector{NumOrArray}(undef, t1+1)

        for t2 = 1:t1+1
            newok, newos = compute_Ω(hist.gk[t1+1,t2], hist.gs[t1+1,t2])
            new_Ωk[t2] = newok
            new_Ωs[t2] = newos
        end
        
        push!(hist.Ωk, new_Ωk)
        push!(hist.Ωs, new_Ωs)
        
        new_Πk = Vector{NumOrArray}(undef, t1+1)
        new_Πs = Vector{NumOrArray}(undef, t1+1)

        for t2 = 1:t1+1
            newpk, newps = compute_Π(hist.gk[t1+1,t2], hist.gs[t1+1,t2], bp)
            new_Πk[t2] = newpk
            new_Πs[t2] = newps
        end
        
        push!(hist.Πk, new_Πk)
        push!(hist.Πs, new_Πs)
        
        # ---------------------------------------------------------------------------
        # Compute mean-field propagators

        new_Mk = Vector{NumOrArray}(undef, t1+1)
        new_Ms = Vector{NumOrArray}(undef, t1+1)
        
        # Ms needs to be computed first, as it only depends on itself.
        
        for t2 = 1:t1+1
            new_Ms[t2] = compute_Ms(hist, sp, t1+1, t2, tstep)
        end
        
        push!(hist.Ms, new_Ms)
        
        # Mk needs to be computed second as it depends on Ms. However, Mk[t1+1,t1+1]
        # needs to be computed separately after the loop because the integral 
        # required calls all the Mk[t1+1,1:t1].
        
        for t2 = 1:t1
            new_Mk[t2] = compute_Mk(hist, sp, t1+1, t2, tstep)
        end
        
        push!(hist.Mk, new_Mk)
        
        hist.Mk[t1+1,t1+1] = zeros(ComplexF64, 3,3,sp.N,sp.N)
        hist.Mk[t1+1,t1+1] = compute_Mk(hist, sp, t1+1, t1+1, tstep)
        
        new_Dk = Vector{NumOrArray}(undef, t1+1)
        new_Ds = Vector{NumOrArray}(undef, t1+1)
        
        for t2 = 1:t1+1
            new_Ds[t2] = compute_Ds(hist, bp, t1+1, t2, tstep)
        end
        
        push!(hist.Ds, new_Ds)
        
        for t2 = 1:t1
            new_Dk[t2] = compute_Dk(hist, bp, t1+1, t2, tstep)
        end
        
        push!(hist.Dk, new_Dk)
        
        hist.Dk[t1+1,t1+1] = zeros(ComplexF64, 3,3,sp.N,sp.N)
        hist.Dk[t1+1,t1+1] = compute_Dk(hist, bp, t1+1, t1+1, tstep)
        
        # ---------------------------------------------------------------------------
        # Compute spin self-energies
        
        new_Σk = Vector{NumOrArray}(undef, t1+1)
        new_Σs = Vector{NumOrArray}(undef, t1+1)
        
        for t2 = 1:t1+1
            newsk, newss = compute_Σ(hist.gk[t1+1,t2],hist.gs[t1+1,t2],
                hist.Mk[t1+1,t2],hist.Ms[t1+1,t2],
                hist.Dk[t1+1,t2],hist.Ds[t1+1,t2], sp, bp)
            new_Σk[t2] = newsk
            new_Σs[t2] = newss
        end
        
        push!(hist.Σk, new_Σk)
        push!(hist.Σs, new_Σs)
        
        # ---------------------------------------------------------------------------
        # Start of the correction phase
        
        for rep = 1:1
            compute_hamiltonian!(ham, hist, sp, bp, t1+1)

            ∂gk_v1 = Vector{NumOrArray}(undef, t1)
            ∂gs_v1 = Vector{NumOrArray}(undef, t1)
            for t2 = 1:t1
                dkv1, dsv1 = rhs_vertical(hist, ham, t1+1, t2, tstep)
                ∂gk_v1[t2] = dkv1
                ∂gs_v1[t2] = dsv1
            end

            ∂gk_d1 = rhs_diag(hist, ham, t1+1, tstep)

            for t2=1:t1
                hist.gk[t1+1,t2] = hist.gk[t1,t2] + 0.5*tstep*(∂gk_v[t2] + ∂gk_v1[t2])
                hist.gs[t1+1,t2] = hist.gs[t1,t2] + 0.5*tstep*(∂gs_v[t2] + ∂gs_v1[t2]);
            end

            hist.gk[t1+1,t1+1] = hist.gk[t1,t1] + 0.5*tstep*(∂gk_d + ∂gk_d1)

            hist.Λ[t1+1] = compute_Λ(hist.gk[t1+1,t1+1], sp)
            hist.λ[t1+1] = compute_λ(hist, bp, t1+1, tstep)

            for t2 = 1:t1+1
                newok, newos = compute_Ω(hist.gk[t1+1,t2], hist.gs[t1+1,t2])
                hist.Ωk[t1+1,t2] = newok
                hist.Ωs[t1+1,t2] = newos
                
                newpk, newps = compute_Π(hist.gk[t1+1,t2], hist.gs[t1+1,t2], bp)
                hist.Πk[t1+1,t2] = newpk
                hist.Πs[t1+1,t2] = newps
            end
            
            
            for t2 = 1:t1+1
                hist.Ms[t1+1,t2] = compute_Ms(hist, sp, t1+1, t2, tstep)
                hist.Ds[t1+1,t2] = compute_Ds(hist, bp, t1+1, t2, tstep)
            end

            for t2 = 1:t1+1
                hist.Mk[t1+1,t2] = compute_Mk(hist, sp, t1+1, t2, tstep)
                hist.Dk[t1+1,t2] = compute_Dk(hist, bp, t1+1, t2, tstep)
            end
            
            for t2 = 1:t1+1
                newsk, newss = compute_Σ(hist.gk[t1+1,t2],hist.gs[t1+1,t2],
                    hist.Mk[t1+1,t2],hist.Ms[t1+1,t2],
                    hist.Dk[t1+1,t2],hist.Ds[t1+1,t2], sp, bp)
                hist.Σk[t1+1,t2] = newsk
                hist.Σs[t1+1,t2] = newss
            end
        end
    end
    append!(hist.times, new_times)
end


