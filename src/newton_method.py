import numpy
from matplotlib import pyplot
import localization

def logistic_regression(data_0=None, data_1=None, mean0=None, mean1=None, cov=None, size=1000, eta=1.0, lam=0.0, precision=1e-6, dim=2):
    """
    Logistic regression using Newton's method.
    """
    print('Newton\'s method with mean0 = ', mean0,
        'mean1 = ', mean1,
        'cov = ', cov,
        'size = ', size,
        'eta = ', eta,
        'lam = ', lam,
        'precision = ', precision,
        'is calculating...'
    )

    if(type(data_0) == type(None)):
        # See equation (22), m = size
        Y0_X = numpy.random.multivariate_normal(mean0, cov, size)
        Y1_X = numpy.random.multivariate_normal(mean1, cov, size)
    else:
        Y0_X = data_0
        Y1_X = data_1

    # Predict function, see equation (14)
    def h_theta(theta, x):
        return 1.0/(1+numpy.exp(numpy.dot(-theta, x)))
    # First derivative, see equation (42)
    def first_derivative(theta):
        res = numpy.zeros(dim+1)
        for i in range(size):
            res += numpy.insert(Y0_X[i], 0, 1.0) * (0.0 - h_theta(theta, numpy.insert(Y0_X[i], 0, 1.0)))
            res += numpy.insert(Y1_X[i], 0, 1.0) * (1.0 - h_theta(theta, numpy.insert(Y1_X[i], 0, 1.0)))
        return res
    # Second derivative Hassian matrix, see equation (31) (32) (33)
    def hassian_matrix(theta):
        # Convert data to equation (31)
        all_x = numpy.ones((1, size*2))
        X = numpy.transpose(numpy.append(Y0_X, Y1_X, 0))
        all_x = numpy.append(all_x, X, 0)
        
        # Calculate A, see equation (33)
        A = numpy.zeros((size*2, size*2))
        for i in range(size):
            h = h_theta(theta, numpy.insert(Y0_X[i], 0, 1.0))
            A[i][i] = h*(h-1)
            h = h_theta(theta, numpy.insert(Y1_X[i], 0, 1.0))
            A[i+size][i+size] = h*(h-1)

        return numpy.matmul(
            all_x,
            numpy.matmul(A, numpy.transpose(all_x))
        )
    
    cur_theta = numpy.zeros(dim+1)
    previous_step_size = 1e8
    iters = 0
    # Iteration, see equation (40)
    while previous_step_size > precision:
        learning = numpy.matmul(
            numpy.linalg.inv(hassian_matrix(cur_theta)),
            first_derivative(cur_theta)
        )
        current_step_size = numpy.linalg.norm(learning)
        print('Current step size:', current_step_size)

        if iters >=1:    
            if(current_step_size > previous_step_size):
                eta /= 10
            # elif(numpy.abs((current_step_size - previous_step_size))/previous_step_size < 0.1):
            #     eta *= 5
        # print('Current eta', eta)

        previous_step_size = current_step_size
        cur_theta -= eta * learning
        iters+=1
    
    pyplot.title(f't = sin(2$\\pi x$)\n'
        f'逻辑回归-牛顿法 \n N = {size}\n'
        f'mean0 = {mean0}, mean1 = {mean1}, cov = {cov},\n 学习率$ \\eta$ = {eta}, 截止步长 = {precision}, 正则项 $\\lambda$ = {lam}\n'
        f'迭代次数: {iters} 次')

    def cal_error_rate():
        total = size*2
        error = 0
        for i in range(size):
            if(numpy.dot(cur_theta, numpy.insert(Y0_X[i], 0, 1.0)) >= 0):
                error += 1
            if(numpy.dot(cur_theta, numpy.insert(Y1_X[i], 0, 1.0)) <= 0):
                error += 1
        return error / total

    # Save theta
    with open(f'../training_results/newton-method-{size}-{lam}.txt', 'w+') as training_results:
        training_results.write(f'[theta_0 theta_1 theta_2] = \n\t' + str(cur_theta.reshape(-1)) + '\n\n')
        training_results.write(f'error rate = ' + str(cal_error_rate()))

    # Border function, see equation (12)
    if mean0==None:
        borderX_1 = numpy.arange(1, 10, 0.01)
    else:
        borderX_1 = numpy.arange(numpy.min([mean0[0], mean1[0]])*0.01, numpy.max([mean0[0], mean1[0]])*2.5, 0.01)
    borderX_2 = (-cur_theta[0]-cur_theta[1]*borderX_1)/cur_theta[2]
    # Plot the border curve
    pyplot.plot(borderX_1, borderX_2, 'g')
    # Plot points
    pyplot.plot(Y0_X[:, 0], Y0_X[:, 1], 'ro')
    pyplot.plot(Y1_X[:, 0], Y1_X[:, 1], 'bo')

    # Save to /images
    pyplot.savefig(f'../images/newton-method-{size}-{lam}.png', bbox_inches='tight')
    pyplot.close()
    print(f'Done! iteration times: {iters}')

if __name__ == "__main__":
    # Case 1:
    logistic_regression(
        None, None,
        (0.8, 1.0),
        (1.4, 1.6), [[0.2, 0.0], [0.0, 0.3]],
        size=20, 
        eta=1,
        lam=0.0,
        precision=0.001
    );

    # Case 2: with regularization
    logistic_regression(
        None, None,
        (0.8, 1.0),
        (1.4, 1.6), [[0.2, 0.0], [0.0, 0.3]],
        size=20, 
        eta=1,
        lam=1.2,
        precision=0.001
    );

    # Case 3:
    logistic_regression(
        None, None,
        (0.8, 1.0),
        (1.4, 1.6), [[0.2, 0.0], [0.0, 0.3]],
        size=200, 
        eta=1,
        lam=0.0,
        precision=0.001
    );

    # Case 4:
    logistic_regression(
        None, None,
        (0.8, 1.0),
        (1.4, 1.6), [[0.2, 0.0], [0.0, 0.3]],
        size=200, 
        eta=1,
        lam=1.2,
        precision=0.001
    );

    # Case 5: larger means
    logistic_regression(
        None, None,
        (30, 40),
        (70, 80), [[1, 0.0], [0.0, 5]],
        size=400, 
        eta=2,
        lam=0.0,
        precision=3
    );

    # Case 6: larger means
    logistic_regression(
        None, None,
        (30, 40),
        (70, 80), [[1, 0.0], [0.0, 5]],
        size=400, 
        eta=2,
        lam=1.2,
        precision=3
    );