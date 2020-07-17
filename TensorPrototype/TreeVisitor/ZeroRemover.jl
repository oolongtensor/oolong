include("../Operations.jl")

function removezeroaddition(add::AddOperation)
    new_children = tuple([node for node in add.children if !(node isa ZeroTensor)]...)
    if new_children == add.children
        new_add = add
    else
        new_add = AddOperation(add.shape, new_children, add.freeindices)
    end
    return removezeroaddition(new_add)
end
