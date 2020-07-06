include("Operations.jl")

function solve(add::AddOperation)
    sum = fill(0, add.shape)
    for node in add.children
        sum += node.value
    end
    return sum
end
