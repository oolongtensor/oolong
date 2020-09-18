"""
    SineOperation(A::AbstractTensor{rank}) where rank

Represents computing sin of A.
"""
struct SineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function SineOperation(A::AbstractTensor{rank}) where rank
        new{rank}(A.shape, (A,), ())
    end
end

"""
    Base.sin(s::AbstractTensor{0})

Returns a [`SineOperation`](@ref) represeting sine of s in radians.
"""
function Base.sin(s::AbstractTensor{0})
    return SineOperation(Tensor(s))
end

"""
    CosineOperation(A::AbstractTensor{rank}) where rank

Represents computing cos of A.
"""
struct CosineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function CosineOperation(A::AbstractTensor{rank}) where rank
        new{rank}(A.shape, (A,), ())
    end
end

"""
    Base.cos(s::AbstractTensor{0})

Returns a [`CosineOperation`](@ref) represeting cosine of s in radians.
"""
function Base.cos(s::AbstractTensor{0})
    return CosineOperation(Tensor(s))
end

"""
    TangentOperation(A::AbstractTensor{rank}) where rank

Represents computing tan of A.
"""
struct TangentOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function TangentOperation(A::AbstractTensor{rank}) where rank
        new{rank}(A.shape, (A,),())
    end
end

"""
    Base.tan(s::AbstractTensor{0})

Returns a [`TangentOperation`](@ref) represeting tangent of s in radians.
"""
function Base.tan(s::AbstractTensor{0})
    return TangentOperation(Tensor(s))
end

"""
    ArcsineOperation(A::AbstractTensor{rank}) where rank

Represents computing arcsin of A.
"""
struct ArcsineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function ArcsineOperation(A::AbstractTensor{rank}) where rank
        if A isa ConstantTensor{T} where T<: Number && abs(A.value) > 1
            throw(DomainError(A, " asin is not defined for |x| > 1."))
        end
        new{rank}(A.shape, (A,),())
    end
end

"""
    Base.asin(s::AbstractTensor{0})

Returns a [`ArcsineOperation`](@ref) represeting arcsine of s.
"""
function Base.asin(s::AbstractTensor{0})
    return ArcsineOperation(Tensor(s))
end

"""
    CosineOperation(A::AbstractTensor{rank}) where rank

Represents computing cos of A.
"""
struct ArccosineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function ArccosineOperation(A::AbstractTensor{rank}) where rank
        if A isa ConstantTensor{T} where T<: Number && abs(A.value) > 1
            throw(DomainError(A, " acos is not defined for |x| > 1."))
        end
        new{rank}(A.shape, (A,),())
    end
end

"""
    Base.acos(s::AbstractTensor{0})

Returns a [`ArccosineOperation`](@ref) represeting arccosine of s.
"""
function Base.acos(s::AbstractTensor{0})
    return ArccosineOperation(Tensor(s))
end

"""
    ArctangentOperation(A::AbstractTensor{rank}) where rank

Represents computing arctan of A.
"""
struct ArctangentOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function ArctangentOperation(A::AbstractTensor{rank}) where rank
        new{rank}(A.shape, (A,),())
    end
end

"""
    Base.atan(s::AbstractTensor{0})

Returns a [`ArctangentOperation`](@ref) represeting arctangent of s.
"""
function Base.atan(s::AbstractTensor{0})
    return ArctangentOperation(Tensor(s))
end