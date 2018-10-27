import numpy
from matplotlib import pyplot
import localization

def logistic_regression(mean0, mean1, cov, size=1000, eta=1.0, lam=0.0, precision=1e-6):
    """
    Logistic regression using Gradient Ascent.
    """
    print('Gradient ascent with mean0 = ', mean0,
        'mean1 = ', mean1,
        'cov = ', cov,
        'size = ', size,
        'eta = ', eta,
        'lam = ', lam,
        'precision = ', precision,
        'is calculating...'
    )

    # See equation (22), m = size
    Y0_X = numpy.random.multivariate_normal(mean0, cov, size)
    Y1_X = numpy.random.multivariate_normal(mean1, cov, size)

    # Gradient function, see equation (28)
    def gradient(theta):
        sum = numpy.zeros(3)
        for i in range(size):
            x = numpy.insert(Y0_X[i], 0, 1.0)
            exp = numpy.exp(numpy.dot(-theta, x))
            sum += x*(0.0 - 1.0/(1+exp))

            x = numpy.insert(Y1_X[i], 0, 1.0)
            exp = numpy.exp(numpy.dot(-theta, x))
            sum += x*(1.0 - 1.0/(1+exp))
        return sum - lam*theta

    cur_theta = numpy.zeros(3)
    previous_step_size = 1e8
    iters = 0
    while previous_step_size > precision:
        gdt = gradient(cur_theta)
        current_step_size = numpy.linalg.norm(gdt)
        print('Current gradient:', gdt)

        if iters >=1:    
            if(current_step_size > previous_step_size):
                eta /= 10
            # elif(numpy.abs((current_step_size - previous_step_size))/previous_step_size < 0.1):
            #     eta *= 5
        # print('Current eta', eta)

        previous_step_size = current_step_size
        cur_theta += eta * gdt
        iters+=1
    
    pyplot.title(f't = sin(2$\\pi x$)\n'
        f'逻辑回归-梯度下降法 \n N = {size}\n'
        f'mean0 = {mean0}, mean1 = {mean1}, cov = {cov},\n 学习率$ \\eta$ = {eta}, 截止步长 = {precision}, 正则项 $\\lambda$ = {lam}\n'
        f'迭代次数: {iters} 次')

    # Save theta
    with open(f'../training_results/gradient-ascent-{size}-{lam}.txt', 'w+') as training_results:
        training_results.write(f'[theta_0 theta_1 theta_2] = \n\t' + str(cur_theta.reshape(-1)) + '\n\n')

    # Border function, see equation (12)
    borderX_1 = numpy.arange(numpy.min([mean0[0], mean1[0]])*0.01, numpy.max([mean0[0], mean1[0]])*2.5, 0.01)
    borderX_2 = (-cur_theta[0]-cur_theta[1]*borderX_1)/cur_theta[2]
    # Plot the border curve
    pyplot.plot(borderX_1, borderX_2, 'g')
    # Plot points
    pyplot.plot(Y0_X[:, 0], Y0_X[:, 1], 'ro')
    pyplot.plot(Y1_X[:, 0], Y1_X[:, 1], 'bo')

    # Save to /images
    pyplot.savefig(f'../images/gradient-ascent-{size}-{lam}.png', bbox_inches='tight')
    pyplot.close()
    print(f'Done! iteration times: {iters}')

# Case 1:
logistic_regression(
    (0.8, 1.0),
    (1.4, 1.6), [[0.2, 0.0], [0.0, 0.3]],
    size=20, 
    eta=0.02,
    lam=0.0,
    precision=0.001
);

# Case 2: with regularization
logistic_regression(
    (0.8, 1.0),
    (1.4, 1.6), [[0.2, 0.0], [0.0, 0.3]],
    size=20, 
    eta=0.02,
    lam=1.2,
    precision=0.001
);

logistic_regression(
    (0.8, 1.0),
    (1.4, 1.6), [[0.2, 0.0], [0.0, 0.3]],
    size=200, 
    eta=0.02,
    lam=0.0,
    precision=0.001
);

logistic_regression(
    (0.8, 1.0),
    (1.4, 1.6), [[0.2, 0.0], [0.0, 0.3]],
    size=200, 
    eta=0.02,
    lam=1.2,
    precision=0.001
);
