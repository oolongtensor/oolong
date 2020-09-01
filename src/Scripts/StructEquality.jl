struct ImmutableX
    x
end

mutable struct MutableX
    x
end

counter = 0

struct CounterX
    x
    id
    function CounterX(x)
        global counter
        counter += 1
        new(x, counter)
    end
end


a = ImmutableX(1)
b = ImmutableX(1)
a_mutable = MutableX(1)
b_mutable = MutableX(1)
a_array = ImmutableX([1])
b_array = ImmutableX([1])
a_counter = CounterX(1)
b_counter = CounterX(1)

println(a === b) # True
println(a_mutable === b_mutable) #False
println(a_array === b_array) # False
println(a_counter === b_counter) # False
println(a_counter == b_counter) # False
