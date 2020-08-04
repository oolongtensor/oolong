include("Trigonometry.jl")

import Base

"""Differentiates A w.r.t y"""
function Base.diff(A::Union{Tensor{T, 0}, ConstantTensor{T, 0}}, y::ScalarVariable) where T
    if A.value == y || A.value == [y]
        return ConstantTensor(1)
    else
        return ZeroTensor()
    end
end

"""Differentiates A w.r.t y"""
function Base.diff(Z::ZeroTensor{0}, y::ScalarVariable)
    return Z
end

"""Differentiates A w.r.t y"""
function Base.diff(add::AddOperation{0}, y::ScalarVariable)
    return +([Base.diff(child, y) for child in add.children]...)
end

"""Differentiates A w.r.t y"""
function Base.diff(op::OuterProductOperation{0}, y::ScalarVariable)
    (A, B) = op.children
    return A*Base.diff(B, y) + Base.diff(A, y)*B
end

"""Differentiates A w.r.t y"""
function Base.diff(si::SineOperation{0}, y::ScalarVariable)
    return Base.cos(si.children[1]) * Base.diff(si.children[1], y)
end

"""Differentiates A w.r.t y"""
function Base.diff(co::CosineOperation{0}, y::ScalarVariable)
    return - Base.sin(co.children[1]) * Base.diff(co.children[1], y)
end