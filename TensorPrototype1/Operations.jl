abstract type Operation end

# Should I instead have a single operation type with the operation type defined?
struct Add <: Operation
    children::Tuple{Vararg{Node}}
end
