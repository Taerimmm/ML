def getKneightCorridinates(location):
    coordinates = []
    x =  location[1]
    y = location [0]
    x_line = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    y_line = ['1', '2', '3', '4', '5', '6', '7', '8']
    x_idx = x_line.index(x)
    y_idx = y_line.index(y)

    index = []
    index.append([x_idx + 2, y_idx+1 ])
    index.append([x_idx + 1, y_idx+2 ])
    index.append([x_idx - 2, y_idx+1 ])
    index.append([x_idx - 1, y_idx+2 ])
    index.append([x_idx + 2, y_idx-1 ])
    index.append([x_idx + 1, y_idx-2 ])
    index.append([x_idx - 2, y_idx-1 ])
    index.append([x_idx - 1, y_idx-2 ])

    for i in index:
        if (i[0] < 0) or (i[0] > 7):
            continue
        elif (i[1] <0) or (i[1] > 7):
            continue
        else:
            coordinates.append(y_line[i[1]] + x_line[i[0]])

    coordinates.sort()
    return coordinates

print(getKneightCorridinates('1e'))