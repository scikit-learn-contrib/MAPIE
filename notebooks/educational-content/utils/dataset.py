import numpy as np


def x_sinx(x):
    """
    One-dimensional x*sin(x) function.
    """
    return x*np.sin(x)


def get_1d_data_with_constant_noise(funct, min_x, max_x, n_samples, noise):
    """
    Generate 1D noisy data uniformely from the given function 
    and standard deviation for the noise.
    """
    generator = np.random.RandomState(59)
    X_train = np.linspace(min_x, max_x, n_samples)
    np.random.shuffle(X_train)
    X_test = np.linspace(min_x, max_x, n_samples*5)
    y_train, y_mesh, y_test = funct(X_train), funct(X_test), funct(X_test)
    y_train += generator.normal(0, noise, y_train.shape[0])
    y_test += generator.normal(0, noise, y_test.shape[0])
    return (
        X_train.reshape(-1, 1),
        y_train, X_test.reshape(-1, 1),
        y_test,
        y_mesh
    )


def get_1d_data_with_heteroscedastic_noise(
    funct,
    min_x,
    max_x,
    n_samples,
    noise
):
    """
    Generate 1D noisy data uniformely from the given function 
    and standard deviation for the noise.
    """
    generator = np.random.RandomState(59)
    X_train = np.linspace(min_x, max_x, n_samples)
    np.random.shuffle(X_train)
    X_test = np.linspace(min_x, max_x, n_samples*5)
    y_train, y_mesh, y_test = funct(X_train), funct(X_test), funct(X_test)
    y_train += (generator.normal(0, noise, len(X_train)) * X_train)
    y_test += (generator.normal(0, noise, len(X_test)) * X_test)
    return (
        X_train.reshape(-1, 1),
        y_train,
        X_test.reshape(-1, 1),
        y_test,
        y_mesh
    )


def get_1d_data_with_normal_distribution(
    funct,
    mu,
    sigma,
    n_samples,
    noise
):
    """
    Generate noisy 1D data with normal distribution from given function 
    and noise standard deviation.
    """
    generator = np.random.RandomState(59)
    X_train = generator.normal(mu, sigma, n_samples)
    X_test = np.sort(generator.normal(mu, sigma, n_samples*5))
    y_train, y_mesh, y_test = funct(X_train), funct(X_test), funct(X_test)
    y_train += generator.normal(0, noise, y_train.shape[0])
    y_test += generator.normal(0, noise, y_test.shape[0])
    return (
        X_train.reshape(-1, 1),
        y_train,
        X_test.reshape(-1, 1),
        y_test,
        y_mesh
    )


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def sigmoid_over(x):
    y = sigmoid(x + 2)
    return y


def sigmoid_under(x):
    y = sigmoid(x - 2)
    return y


def sigmoid_mixed_1(x):
    y = sigmoid(2*x)
    return y


def sigmoid_mixed_2(x):
    y = sigmoid(0.5*x)
    return y


def generate_y_true_calibrated(y_prob):
    uniform = np.random.uniform(size=len(y_prob))
    y_true = (uniform <= y_prob).astype(float)
    return y_true
