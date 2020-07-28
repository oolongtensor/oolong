abstract type Node end

counter = 0

function getcounter()
    global counter
    counter += 1
    return counter
end

function getnumberofnodes()
    global counter
    return counter
end
