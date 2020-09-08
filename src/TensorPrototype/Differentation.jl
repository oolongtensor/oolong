"""Differentiates A w.r.t y"""
function _differentiate(A::Union{Tensor{T, 0}, ConstantTensor{T, 0}}, y::VariableTensor{0}, difffn) where T
    return A.value == y || A.value == [y] ? ConstantTensor(1) : ZeroTensor()
end

function _differentiate(A::VariableTensor{0}, y::VariableTensor{0}, difffn)
    return A == y ? ConstantTensor(1) : ZeroTensor()
end

function _differentiate(Z::ZeroTensor{0}, y::VariableTensor{0}, difffn)
    return Z
end

function _differentiate(add::AddOperation{0}, y::VariableTensor{0}, difffn)
    return +([difffn(child) for child in add.children]...)
end

function _differentiate(op::OuterProductOperation{0}, y::VariableTensor{0}, difffn)
    A, B = op.children
    return A*difffn(B) + difffn(A)*B
end

function _differentiate(si::SineOperation{0}, y::VariableTensor{0}, difffn)
    return Base.cos(si.children[1]) * difffn(si.children[1])
end

function _differentiate(co::CosineOperation{0}, y::VariableTensor{0}, difffn)
    return - Base.sin(co.children[1]) * difffn(co.children[1])
end

function _differentiate(root::RootNode, y::VariableTensor{0}, diffn)
    return RootNode(diffn(root.children[1]))
end

"""Differentiates A w.r.t y"""
function differentiate(A::AbstractTensor{0}, x::VariableTensor{0})
    return traversal(A, x -> x, _differentiate, nothing, x, nothing, true)
end