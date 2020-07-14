import Base

abstract type Node end

function Base.show(io::IO, node::Node, depth::Int)
    print(io, ["\t" for i in 1:depth]..., typeof(node), " {")
    for name in fieldnames(typeof(node))
        fieldcontent = getfield(node, name)
        if fieldcontent != () && name != :children
            print(io, " ", name, " : ", fieldcontent, ",")
        end
    end
    print(io, "}\n")
    for child in node.children
        Base.show(io, child, depth + 1)
    end
end

Base.show(io::IO, node::Node) = Base.show(io, node, 0)
