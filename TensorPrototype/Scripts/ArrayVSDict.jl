function tensorvsmatrix(size)
    arraysize = Base.summarysize(fill(0, (size)))
    dict = Dict()
    if Base.summarysize(dict) > arraysize
        return -1
    end
    for i in 1:size
        dict[i] = i*1.0
        if Base.summarysize(dict) > arraysize
            return i
        end
    end
    return size + 1
end

println("Start")

x = 10
while x < 100001
    global x
    println((x, tensorvsmatrix(x)))
    x *= 10
end
