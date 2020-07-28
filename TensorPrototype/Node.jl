abstract type Node end

_counter = 0

function getcounter()
    global _counter
    _counter += 1
    return _counter
end

function getnumberofnodes()
    global _counter
    return _counter
end

# Two nodes are equal if everything but their indices are equal
function Base.:(==)(a::Node, b::Node)
    if a === b
        return true
    end
    if typeof(a) != typeof(b)
        return false
    end
    for name in fieldnames(typeof(a))
        if name != :index && getproperty(a, name) != getproperty(b, name)
            return false
        end
    end
    return true
end
