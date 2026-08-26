## What Activation Functions Do

A neural network is built from layers of linear transformations: multiply by a weight matrix, add a bias. The problem is that stacking linear functions just gives you another linear function. Two layers with no activation:

$$
y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)
$$

This is still just $y = Ax + c$, a single linear function. No matter how many layers you stack, the network can only learn linear relationships. It could never learn to classify images, understand language, or approximate complex patterns.

**Activation functions** break this linearity. After each linear transformation, an activation function applies a nonlinear operation element-wise. This is what gives neural networks their power to approximate any continuous function.

---

## ReLU: The Simplest Nonlinearity

ReLU (Rectified Linear Unit) is defined as:

$$
\text{ReLU}(x) = \max(0, x)
$$

The rule is extremely simple:

- If $x > 0$: output $x$ (pass it through unchanged)
- If $x \leq 0$: output $0$ (block it)

Some examples:

- $\text{ReLU}(3.5) = 3.5$
- $\text{ReLU}(0) = 0$
- $\text{ReLU}(-2.7) = 0$
- $\text{ReLU}(100) = 100$

The function looks like a hockey stick: flat at zero for all negative inputs, then a straight line with slope 1 for positive inputs.

---

## Why ReLU Became Dominant

Before ReLU, the standard activations were **sigmoid** and **tanh**. Both squash their inputs into a bounded range (sigmoid to $(0,1)$, tanh to $(-1,1)$). They worked, but they had a serious problem: **vanishing gradients**.

The vanishing gradient problem:

- Sigmoid and tanh saturate for large inputs (the output barely changes)
- In the saturated regions, the derivative is nearly zero
- During backpropagation, gradients get multiplied through many layers
- Near-zero derivatives at each layer cause the gradient to shrink exponentially
- Deep layers barely learn because their gradient signal is essentially zero

ReLU fixes this for positive inputs:

- The derivative of ReLU is 1 for all $x > 0$
- Gradients pass through unchanged, no matter how many layers
- Deep networks can train much faster because gradient signals survive

ReLU also has practical advantages:

- **Computationally cheap**: just a comparison with zero, no exponentials or divisions
- **Sparse activation**: many neurons output exactly zero, which creates sparse representations. Sparse networks are more efficient and often generalize better.
- **Easy to implement**: one line of code

---

## The Gradient of ReLU

The derivative (used during backpropagation) is:

$$
\frac{d}{dx} \text{ReLU}(x) =
\begin{cases}
    1, & \text{if } x > 0 \\
    0, & \text{if } x < 0
\end{cases}
$$

At $x = 0$, the function has a sharp corner and is technically not differentiable. In practice, frameworks define the derivative at zero as 0 (some use 0.5 or 1). This does not cause problems in practice because hitting exactly $x = 0$ is extremely rare with floating-point numbers.

The gradient being exactly 1 for positive inputs is what makes ReLU so effective for deep networks. Compare this to sigmoid, where the maximum gradient is only 0.25 (at $x = 0$). After 10 layers of sigmoid, the gradient shrinks by a factor of $0.25^{10} \approx 10^{-6}$.

---

## The Dead Neuron Problem

ReLU's main weakness: for negative inputs, the output and gradient are both exactly zero. If a neuron's weighted input is negative for every training example, it will never produce a nonzero output and never receive a nonzero gradient. It is permanently "dead."

How neurons die:

- A large gradient update pushes the weights so that the pre-activation is always negative
- Once negative for all inputs, the gradient is always zero, so the weights never update again
- The neuron contributes nothing to the network for the rest of training

This can happen to a significant fraction of neurons, especially with high learning rates. In some networks, 10-40% of neurons can die.

Solutions that build on ReLU:

- **Leaky ReLU**: allows a small gradient for negative inputs ($\alpha x$ instead of $0$)
- **ELU**: uses an exponential curve for negative inputs
- **GELU/Swish**: smooth approximations that never have exactly zero gradient
- **Careful initialization**: initializing weights properly (e.g., He initialization) reduces the chance of neurons starting in the dead zone

---

## Where ReLU Is Used

ReLU is the **default activation** for most neural network architectures:

- **Convolutional networks (CNNs)**: AlexNet (2012) was one of the first major successes using ReLU, and every major CNN since has used it or a variant
- **Fully connected layers**: the standard choice for hidden layers in MLPs
- **Generative models**: used in generators and discriminators of GANs
- **Residual networks (ResNets)**: ReLU after each residual block

ReLU is less common in:
- **Output layers**: where you need bounded outputs (use sigmoid or softmax instead)
- **Transformers**: which typically use GELU or Swish in the feed-forward layers
- **RNNs**: which use tanh or sigmoid for the recurrent connections (ReLU can cause exploding activations in recurrent loops)