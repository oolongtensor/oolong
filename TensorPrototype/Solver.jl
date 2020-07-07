include("Operations.jl")

function solve(add::AddOperation)
    sum = fill(0, add.shape)
    for node in add.children
        sum += solve(node)
    end
    return sum
end

function solve(tensor::ConcreteTensor)
    return tensor.value
end

function solve(trans::TransposeOperation)
    return transpose(solve(trans.children[1]))
end
