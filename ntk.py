import numpy as np
import jax.numpy as jnp
import jax.scipy as jsci
import matplotlib.pyplot as plt
from jax import grad, vmap, jit

def relu(x):
    return jnp.maximum(x,0)

@jit
def predict(params, inputs):
  activations = inputs
  for w in params[:-1]:
    outputs = jnp.dot(w, activations)
    activations = relu(outputs)
  net_out = jnp.dot(params[-1], activations)
  return net_out

@jit
def predict_from_flat(params_flat,inputs):
  layer_break_idxs = range(len(layer_breaks))
  params = [params_flat[layer_breaks[i]:layer_breaks[j]].reshape(layer_sizes[j],layer_sizes[i])
            for i,j in zip(layer_break_idxs[:-1],layer_break_idxs[1:])]
  activations = inputs
  for w in params[:-1]:
    outputs = jnp.dot(w, activations)
    activations = relu(outputs)
  net_out = jnp.dot(params[-1], activations)
  if net_out.shape == (1,):
      return net_out[0]
  else:
      return net_out
@jit
def output_neuron_grad(inputs):
  neuron_grad = grad(predict_from_flat)(params_flat,inputs)
  return neuron_grad

@jit
def loss(params, batch):
  inputs, targets = batch
  net_out = predict(params, inputs)
  return 0.5*jnp.power(jnp.linalg.norm(net_out - targets),2)

def init_random_params(scale, layer_sizes, seed):
  # Returns a list of tuples where each tuple is the weight matrix and bias vector for a layer
  np.random.seed(seed)
  return [np.random.normal(0.0, scale/np.sqrt(n), (n, m)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

layer_sizes = [2,5000,1]
lr = 1e-5
num_epochs = 20
ts = np.arange(num_epochs+1)
closed_form_errors = np.zeros(num_epochs+1)
simulation_errors = np.zeros(num_epochs)

params = init_random_params(1.0, layer_sizes, np.random.randint(10000000))
params_flat = np.hstack([params[i].reshape(-1) for i in range(len(params))])
layer_breaks = np.hstack([0,np.cumsum([i*j for i,j in zip(layer_sizes[:-1], layer_sizes[1:])])]) 

x = np.array([[1.0,1.0,-1.0,-1.0],[1.0,-1.0,-1.0,1.0]])
y = np.array([1.0,-1.0,-1.0,1.0])

# Getting the NTK
num_params = np.sum([i*j for i,j in zip(layer_sizes[:-1], layer_sizes[1:])])
grad_W = np.zeros((num_params, x.shape[1]))
output_grads = np.array(vmap(output_neuron_grad)(x.T))

# Using NTK to predict the dynamics
print("Using NTK to predict network output and error")
ntk = jnp.dot(output_grads,output_grads.T)
init_error = y - predict(params, x)
for i in range(len(ts)):
    t = ts[i]
    model_out = - jnp.dot(init_error, jsci.linalg.expm(-lr*ntk*t)) + y
    diffs = y - model_out
    closed_form_errors[i] = 0.5*jnp.dot(y - model_out, (y - model_out).T)**2
    print("Model out: ", model_out)
    print("Error: ", closed_form_errors[i])

# Now train the network with GD
print("\n Simulate the network training with GD")
@jit
def update(params, batch):
  grads = grad(loss)(params, batch)
  return [w - lr * dw for w,dw in zip(params, grads)]

for i in range(num_epochs):
    params = update(params, (x, y))
    model_out = predict(params,x)
    simulation_errors[i] = 0.5*jnp.dot(y - model_out, (y - model_out).T)**2
    print("Model out: ", model_out)
    print("Error: ", simulation_errors[i])

plt.plot(closed_form_errors[1:], label='Predicted Errors', linestyle='dashed')
plt.plot(simulation_errors, label='Simulated Errors')
#plt.title("Network Loss Over Training Multiple Generations")
plt.ylabel("Error")
plt.xlabel("Epoch number")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.grid()
plt.legend()
plt.savefig("ntk_errors.png")
plt.close()
