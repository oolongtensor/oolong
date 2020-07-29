include("../TreeVisitor.jl")

function removezero!(add::AddOperation)
    # Somehow remove children which are ZeroTensor
end

function removezero!(node::Node)
    return
end
