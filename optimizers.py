import numpy as np


class SGD:
    """
    Implements stochastic gradient descent.
    The optimizer accesses trainable layers of the model (those with attributes
    'w' and 'b') and uses their stored gradients (dw and db) to update parameters.
    """
    def __init__(self, model, loss_func, learning_rate=1e-3, l2_reg=None):
        self.model = model
        self.loss_func = loss_func
        self.lr = learning_rate
        self.reg_strength = None
        if l2_reg:
            self.reg_strength = l2_reg

    def backward(self, y_pred, y_true):
        """
        Given the model predictions and the true values, compute the upstream gradient
        from the loss and backpropagate through the model.
        """
        # Compute loss gradient with respect to predictions.
        dout = self.loss_func.backward(y_pred, y_true)
        # Backpropagate through the model.
        self.model.backward(dout)

    def _update(self, param, grad, lr):
        if self.reg_strength:
            param_updated = param - lr * (grad + 2 * self.reg_strength * param)
        else:
            param_updated = param - lr * grad 

        return param_updated

    def step(self):
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'w'):
                # Ensure that the layer has computed its gradient.
                if layer.dw is not None:
                    updated_param = self._update(layer.w, layer.dw, self.lr)
                    layer.w = updated_param
                    # optional, reset the gradient after update.
                    layer.dw = np.zeros_like(layer.w)
            if hasattr(layer, 'b'):
                if layer.db is not None:
                    updated_param = self._update(layer.b, layer.db, self.lr)
                    layer.b = updated_param
                    layer.db = np.zeros_like(layer.b)



class SGD_Momentum:
    """
    Implements stochastic gradient descent with momentum.
    The optimizer accesses trainable layers of the model (those with attributes
    'w' and 'b') and uses their stored gradients (dw and db) to update parameters.
    """
    def __init__(self, model, loss_func, learning_rate=1e-4, l2_reg=None, **kwargs):
        self.model = model
        self.loss_func = loss_func
        self.lr = learning_rate
        self.reg_strength = None
        if l2_reg:
            self.reg_strength = l2_reg
        # Optimizer configuration; default momentum is 0.9 if not provided.
        self.optim_config = kwargs.pop('optim_config', {'momentum': 0.9})
        # A dictionary to hold velocity for each parameter.
        self.velocities = {}
        self._init_velocities()

    def _init_velocities(self):
        """
        Initializes velocity arrays for all trainable parameters in the model.
        The keys are strings like 'layer0_w', 'layer0_b', etc.
        """
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'w'):
                self.velocities[f'layer{i}_w'] = np.zeros_like(layer.w)
            if hasattr(layer, 'b'):
                self.velocities[f'layer{i}_b'] = np.zeros_like(layer.b)

    def backward(self, y_pred, y_true):
        """
        Given the model predictions and the true values, compute the upstream gradient
        from the loss and backpropagate through the model.
        """
        # Compute loss gradient with respect to predictions.
        dout = self.loss_func.backward(y_pred, y_true)
        # Backpropagate through the model.
        self.model.backward(dout)

    def _update(self, param, grad, velocity, lr):
        """
        Performs a single parameter update using SGD with momentum.
        :param param: The parameter array to update.
        :param grad: The gradient array for the parameter.
        :param velocity: The current velocity for this parameter.
        :param lr: Learning rate.
        :return: The updated parameter and velocity.
        """
        if self.reg_strength:
            grad = grad + 2 * self.reg_strength * grad
        momentum = self.optim_config.get('momentum', 0.9)
        velocity = momentum * velocity - lr * grad
        param_updated = param + velocity
        return param_updated, velocity

    def step(self):
        """
        Update the parameters of the model.
        For each trainable layer (ones with 'w' and/or 'b'), update their parameters
        using the stored gradients.
        """
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'w'):
                key = f'layer{i}_w'
                # Ensure that the layer has computed its gradient.
                if layer.dw is not None:
                    updated_param, new_velocity = self._update(
                        layer.w, layer.dw, self.velocities[key], self.lr)
                    layer.w = updated_param
                    self.velocities[key] = new_velocity
                    # Optionally, reset the gradient after update.
                    layer.dw = np.zeros_like(layer.w)
            if hasattr(layer, 'b'):
                key = f'layer{i}_b'
                if layer.db is not None:
                    updated_param, new_velocity = self._update(
                        layer.b, layer.db, self.velocities[key], self.lr)
                    layer.b = updated_param
                    self.velocities[key] = new_velocity
                    layer.db = np.zeros_like(layer.b)



class Adam:
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format for each parameter:
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    def __init__(self, model, loss_func, learning_rate=1e-4, **kwargs):
        self.model = model
        self.loss_func = loss_func
        self.lr = learning_rate
        self.optim_config = kwargs.pop('optim_config', {})
        self._reset()

    def _reset(self):
        """
        Initialize an optimizer configuration dictionary for each trainable parameter
        in the model. Here we iterate over model.layers and create an entry for
        each parameter (e.g. 'layer0_w', 'layer0_b', etc.).
        """
        self.optim_configs = {}
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'w'):
                key = f'layer{i}_w'
                self.optim_configs[key] = {}
            if hasattr(layer, 'b'):
                key = f'layer{i}_b'
                self.optim_configs[key] = {}

    def backward(self, y_pred, y_true):
        """
        Compute the gradients with respect to the model parameters.
        First compute the loss gradient, then backpropagate through the model.
        """
        dout = self.loss_func.backward(y_pred, y_true)
        self.model.backward(dout)

    def _update(self, param, grad, config, lr):
        """
        Update a single model parameter using Adam.
        :param param: The parameter array to update.
        :param grad: The gradient array for the parameter.
        :param config: A dictionary holding Adam configuration for this parameter.
        :param lr: Learning rate.
        :return: The updated parameter and the updated configuration dictionary.
        """
        if config is None:
            config = {}
        # Set defaults for Adam hyperparameters and state.
        config.setdefault('beta1', 0.9)
        config.setdefault('beta2', 0.999)
        config.setdefault('epsilon', 1e-4)
        config.setdefault('m', np.zeros_like(param))
        config.setdefault('v', np.zeros_like(param))
        config.setdefault('t', 0)

        beta1 = config['beta1']
        beta2 = config['beta2']
        eps = config['epsilon']
        m = config['m']
        v = config['v']
        t = config['t']

        # Update biased first moment estimate.
        m = beta1 * m + (1 - beta1) * grad
        # Update biased second moment estimate.
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        # Compute bias-corrected estimates.
        m_hat = m / (1 - np.power(beta1, t + 1))
        v_hat = v / (1 - np.power(beta2, t + 1))
        # Update parameter.
        param_updated = param - lr * m_hat / (np.sqrt(v_hat) + eps)
        # Increment timestep.
        config['t'] = t + 1
        config['m'] = m
        config['v'] = v

        return param_updated, config

    def step(self):
        """
        Update each trainable parameter in the model using Adam.
        The method iterates over the layers and, for any layer with trainable
        parameters (w and/or b), applies the Adam update.
        """
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'w'):
                key = f'layer{i}_w'
                # Only update if the gradient is not None.
                if layer.dw is not None:
                    updated_param, config = self._update(layer.w, layer.dw, self.optim_configs.get(key), lr=self.lr)
                    layer.w = updated_param
                    self.optim_configs[key] = config
                    # Reset gradient.
                    layer.dw = np.zeros_like(layer.w)
            if hasattr(layer, 'b'):
                key = f'layer{i}_b'
                if layer.db is not None:
                    updated_param, config = self._update(layer.b, layer.db, self.optim_configs.get(key), lr=self.lr)
                    layer.b = updated_param
                    self.optim_configs[key] = config
                    layer.db = np.zeros_like(layer.b)
