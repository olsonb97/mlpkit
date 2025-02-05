class Init:
    def basic(input_size, output_size, range=(-0.1, 0.1), xp=None):
        return xp.random.uniform(range[0], range[1], size=(input_size, output_size))

    def he(input_size, output_size, xp=None):
        std = xp.sqrt(2 / input_size)
        return xp.random.normal(0, std, size=(input_size, output_size))

    def xavier_normal(input_size, output_size, xp=None):
        std = xp.sqrt(1 / (input_size + output_size))
        return xp.random.normal(0, std, size=(input_size, output_size))

    def xavier_uniform(input_size, output_size, xp=None):
        limit = xp.sqrt(6 / (input_size + output_size))
        return xp.random.uniform(-limit, limit, size=(input_size, output_size))

class Activate:
    
    def relu(x, xp=None):
        return xp.maximum(0, x)
    
    def relu_derivative(x, xp=None):
        return xp.where(x > 0, 1, 0)

    def leaky_relu(x, alpha=0.01, xp=None):
        return xp.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(x, alpha=0.01, xp=None):
        return xp.where(x > 0, 1, alpha)

    def tanh(x, xp=None):
        return xp.tanh(x)

    def tanh_derivative(x, xp=None):
        return 1 - xp.tanh(x) ** 2

    def identity(x, xp=None):
        return x

    def identity_derivative(x, xp=None):
        return xp.ones_like(x)

    def softmax(z, xp=None):
        stable_logits = z - xp.max(z, axis=1, keepdims=True)
        numerator = xp.exp(stable_logits)
        denominator = xp.sum(numerator, axis=1, keepdims=True)
        result = numerator / denominator
        return result

    def softmax_derivative(p, xp=None):
        # Not efficient. Best to avoid
        p = p.reshape(-1, 1)
        return xp.diagflat(p) - xp.dot(p, p.T)

class Loss:
    def cross_entropy(p, y, gradient=False, xp=None):
        epsilon = 1e-8
        loss = -xp.mean(xp.sum(y * xp.log(xp.clip(p, epsilon, 1.0)), axis=1))

        if gradient:
            return loss, (p - y) / p.shape[0] # dL_da == dL_dz

        return loss