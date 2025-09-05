using TransmuteDims

"Abstract supertype for all expectation values"
abstract type AbstractExpectationValue end

"Union of Number and Array"
NumOrArray = Union{Number, Array{<: Number}}

"""
SymmetricGreenFunction{R, N} <: AbstractExpectationValue

Type of a symmetric Green function. R is the dimension of the tensor representation of
the GF, and N is the number of single indices it has. For example
g^{ab}_i has R = 3 and N = 1
"""
struct SymmetricGreenFunction{R, N} <: AbstractExpectationValue
    "Ragged list representing lower triangle of time matrix"
    ev::Vector{Vector{<: NumOrArray}}
    "Permutation that encodes symmetry. Single indices are at the end"
    perm::NTuple
    function SymmetricGreenFunction(g::NumOrArray, n::Int)
        r = ndims(g)
        p = collect(1:r)
        for i in 1:2:(r-n-1)
            p[i], p[i+1] = p[i+1], p[i]
        end
        return new{r, n}([[g]], Tuple(p))
    end
end

"""
AntisymmetricGreenFunction{R, N} <: AbstractExpectationValue

Type of a antisymmetric Green function. R is the dimension of the tensor representation of
the GF, and N is the number of single indices it has. For example
g^{ab}_i has R = 3 and N = 1
"""
struct AntisymmetricGreenFunction{R, N} <: AbstractExpectationValue
    "Ragged list representing lower triangle of time matrix"
    ev::Vector{Vector{<: NumOrArray}}
    "Permutation that encodes symmetry. Single indices are at the end"
    perm::NTuple
    function AntisymmetricGreenFunction(g::NumOrArray, n::Int)
        r = ndims(g)
        p = collect(1:r)
        for i in 1:2:(r-n-1)
            p[i], p[i+1] = p[i+1], p[i]
        end
        return new{r, n}([[g]], Tuple(p))
    end
end

"""
AntiHermitianGreenFunction{R, N} <: AbstractExpectationValue

Type of a antisymmetric Green function. R is the dimension of the tensor representation of
the GF, and N is the number of single indices it has. For example
g^{ab}_i has R = 3 and N = 1
"""
struct AntiHermitianGreenFunction{R, N} <: AbstractExpectationValue
    "Ragged list representing lower triangle of time matrix"
    ev::Vector{Vector{<: NumOrArray}}
    "Permutation that encodes symmetry. Single indices are at the end"
    perm::NTuple
    function AntiHermitianGreenFunction(g::NumOrArray, n::Int)
        r = ndims(g)
        p = collect(1:r)
        for i in 1:2:(r-n-1)
            p[i], p[i+1] = p[i+1], p[i]
        end
        return new{r, n}([[g]], Tuple(p))
    end
end


raw"""
SingleField{R, N} <: AbstractExpectationValue

Type of a one-time field. R is the dimension of the tensor representation of
the field, and N is the number of single indices it has. For example
\Lambda^{\alpha}_{ij} has R = 3 and N = 1
"""
struct SingleField{R, N} <: AbstractExpectationValue
    "Vector of fields at different times"
    ev::Vector{<: NumOrArray}
    function SingleField(g::NumOrArray, n::Int)
        r = ndims(g)
        return new{r, n}([g])
    end
    function SingleField(g::Vector{<:NumOrArray}, n::Int)
        r = ndims(g[1])
        return new{r, n}(g)
    end
end

# "x=1 for vertical step, x=2 for diagonal step"
# struct Val{x} end

# Functions to call elements of the time matrix. To get the triangle that is not
# stored, the appropriate generalized transpose operation is applied

function Base.getindex(G::SymmetricGreenFunction, t1::Int, t2::Int)
    if t1>=t2 return G.ev[t1][end-t1+t2]
    else
        # transmute belongs to the TransmuteDims package, and is required because
        # it does not allocate memory, unlike the native permutedims, and instead
        # returns a view of the transposed element.
        return transmute(G.ev[t2][end-t2+t1], G.perm) 
    end
end

function Base.getindex(G::AntisymmetricGreenFunction, t1::Int, t2::Int)
    if t1>=t2 return G.ev[t1][end-t1+t2]
    else return -transmute(G.ev[t2][end-t2+t1], G.perm) end
end

function Base.getindex(G::AntiHermitianGreenFunction, t1::Int, t2::Int)
    if t1 >= t2
        return G.ev[t1][end - t1 + t2]
    else
        # Take conjugate transpose and negate
        return -conj(transmute(G.ev[t2][end - t2 + t1], G.perm))
    end
end

function Base.getindex(F::SingleField, t1::Int)
    return F.ev[t1]
end

function Base.getindex(G::AbstractExpectationValue, t1::Int, t2::Colon)
    return [G[t1,i] for i=1:length(G.ev)]
end

function Base.getindex(G::AbstractExpectationValue, t1::Colon, t2::Int)
    return [G[i,t2] for i=1:length(G.ev)]
end

# Allow for changing contents of an expectation value

function Base.setindex!(G::SymmetricGreenFunction, v::NumOrArray, i::Int, j::Int)
    G.ev[i][j] = v
    return v
end

function Base.setindex!(G::AntisymmetricGreenFunction, v::NumOrArray, i::Int, j::Int)
    G.ev[i][j] = v
    return v
end

function Base.setindex!(G::AntiHermitianGreenFunction, v::NumOrArray, i::Int, j::Int)
    G.ev[i][j] = v
    return v
end

function Base.setindex!(G::SingleField, v::NumOrArray, i::Int)
    G.ev[i] = v
    return v
end

function Base.push!(collection::AbstractExpectationValue, items...)
    push!(collection.ev, items...)
end

function time_diag(G::AbstractExpectationValue)
    return [G[i,i] for i=1:length(G.ev)]
end

