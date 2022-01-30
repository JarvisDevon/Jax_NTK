import numpy as np
import jax.numpy as jnp
import jax.scipy as jsci
import matplotlib.pyplot as plt
from jax import grad, vmap, jit
from datasets import gen_poly_data

@jit
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

layer_sizes = [4,1000000,1]
lr = 1e-9
num_epochs = 400
ts = np.arange(num_epochs+1)
closed_form_errors = np.zeros(num_epochs+1)
simulation_errors = np.zeros(num_epochs)

params = init_random_params(1.0, layer_sizes, np.random.randint(10000000))
params_flat = np.hstack([params[i].reshape(-1) for i in range(len(params))])
layer_breaks = np.hstack([0,np.cumsum([i*j for i,j in zip(layer_sizes[:-1], layer_sizes[1:])])]) 

# Create Dataset
data_shape = (20,4)
inputs_scale = 1.0
gen_params_scale = 1.0
noise_var = 0.1
batch_size = 32
include_bias = True
x, y, _, _, num_batches, batches = \
              gen_poly_data(data_shape, inputs_scale, gen_params_scale, batch_size, noise_var, include_bias, True)

# Getting the NTK
num_params = np.sum([i*j for i,j in zip(layer_sizes[:-1], layer_sizes[1:])])
grad_W = np.zeros((num_params, x.shape[1]))
output_grads = np.array(vmap(output_neuron_grad)(x.T))

# Using NTK to predict the dynamics
print("Using NTK to predict network error")
ntk = jnp.dot(output_grads,output_grads.T)
init_error = y - predict(params, x)
for i in range(len(ts)):
    t = ts[i]
    model_out = - jnp.dot(init_error, jsci.linalg.expm(-lr*ntk*t)) + y
    diffs = y - model_out
    closed_form_errors[i] = (1/data_shape[0])*0.5*jnp.dot(y - model_out, (y - model_out).T)**2
    print("Error: ", closed_form_errors[i])

if include_bias:
    powers_x = data_shape[1] - 1
else:
    powers_x = data_shape[1]
input_density = 500
inputs = np.arange(-3,3,(3--3)/input_density).reshape(input_density,1)
for i in range(powers_x-1):
    inputs = np.hstack([inputs, inputs[:,-1].reshape(input_density,1)*inputs[:,0:1]])
if include_bias:
    inputs = np.hstack([np.ones((input_density,1)), inputs])
#predicted_y_ntk = predict(params,inputs.T) # Need to use inference formula

# Now train the network with GD
print("\n Simulate the network training with GD")
@jit
def update(params, batch):
  grads = grad(loss)(params, batch)
  return [w - lr * dw for w,dw in zip(params, grads)]

for i in range(num_epochs):
    params = update(params, (x, y))
    model_out = predict(params,x)
    simulation_errors[i] = (1/data_shape[0])*0.5*jnp.dot(y - model_out, (y - model_out).T)**2
    print("Error: ", simulation_errors[i])

plt.plot(simulation_errors, label='Simulated Errors')
plt.plot(closed_form_errors[1:], label='Predicted Errors', linestyle='dashed')
#plt.title("Network Loss Over Training Multiple Generations")
plt.ylabel("Error")
plt.ylim(0,5)
plt.xlabel("Epoch number")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.grid()
plt.legend()
plt.savefig("ntk_errors.png")
plt.close()

predicted_y = predict(params,inputs.T)
plt.scatter(x[1].T,y.T, label='Training Data Points')
plt.plot(inputs.T[1],predicted_y.T, label='Network Function')
plt.ylabel("Y")
plt.xlabel("X")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.grid()
plt.legend()
plt.savefig('x_vs_y.png')
plt.close()
