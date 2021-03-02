x_train = 0.5
y_train = 0.8

weight = (x_train / y_train)
lr = 0.01
epoch = 1000

es_count = 0
err = []
for iteration in range(epoch):
    y_predict = x_train * weight
    error = (y_predict - y_train) ** 2

    print("Error : " + str(error) + "\ty_predict : " + str(y_predict))

    up_y_predict = x_train * (weight + lr)
    up_error = (y_train - up_y_predict) ** 2

    down_y_predict = x_train * (weight - lr)
    down_error = (y_train - down_y_predict) ** 2

    if (down_error <= up_error):
        weight = weight - lr
    if (down_error > up_error):
        weight = weight + lr

    es_count += 1

    # Es 걸어보기 !!! 
    if iteration == 0:
        err.append(error)
    
    error_ = (x_train * weight - y_train) ** 2
    
    if error_ < err[-1]:
        err.append(error)

    # Early Stopping
    if iteration - len(err)  == 100:
        print('Epoch :', iteration)
        break