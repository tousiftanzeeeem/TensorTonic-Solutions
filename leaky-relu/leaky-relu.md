## The Dead Neuron Problem

ReLU outputs exactly zero for any negative input:

$$
\text{ReLU}(x) = \max(0, x)
$$

This means if a neuron's pre-activation is negative for every training example, its gradient is always zero. The neuron never updates. It is permanently dead and contributes nothing to the network.

This happens more often than you might expect:

- A large gradient update can push the bias so negative that the pre-activation is always below zero
- With high learning rates, many neurons can die in the first few training steps
- Once dead, there is no way to recover, since the gradient is exactly zero

---

## Leaky ReLU: A Simple Fix

Leaky ReLU modifies the negative side to have a small slope instead of being flat at zero:

$$
\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{if } x < 0 \end{cases}
$$

- For positive inputs, it behaves exactly like ReLU: output equals input
- For negative inputs, it multiplies by a small constant $\alpha$ (typically 0.01)

Some examples with $\alpha = 0.01$:

- $\text{LeakyReLU}(3.0) = 3.0$
- $\text{LeakyReLU}(0) = 0$
- $\text{LeakyReLU}(-2.0) = -0.02$
- $\text{LeakyReLU}(-100) = -1.0$

The "leak" is small but crucial. Instead of completely blocking negative signals, it lets a tiny amount through.

---

## The Role of Alpha

The parameter $\alpha$ controls the slope for negative inputs:

- $\alpha = 0$: this is standard ReLU. No leak at all.
- $\alpha = 0.01$: the most common default. Negative inputs are scaled down by 100x.
- $\alpha = 0.1$ or $0.2$: a larger leak. Used when you want more gradient flow for negative inputs.
- $\alpha = 1$: the function becomes $f(x) = x$ (identity). No activation at all.
- $\alpha > 1$: the negative side has a steeper slope than the positive side. Unusual but valid.

The key insight: **as long as $\alpha \neq 0$, no neuron can ever die.** Even when the pre-activation is negative, there is always a nonzero gradient ($\alpha$), so the weights can still update.

---

## The Gradient

The derivative of Leaky ReLU is:

$$
\frac{d}{dx} \text{LeakyReLU}(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ \alpha & \text{if } x < 0 \end{cases}
$$

Compare this to ReLU, where the derivative for negative inputs is exactly 0. With Leaky ReLU, the derivative for negative inputs is $\alpha$, which is small but nonzero. During backpropagation:

- Positive region: gradient passes through at full strength (multiplied by 1)
- Negative region: gradient is reduced by a factor of $\alpha$ (e.g., 0.01) but still flows

This means gradients can always reach every neuron, regardless of the sign of the input.

---

## Parametric ReLU (PReLU)

A natural extension: instead of fixing $\alpha$ as a constant, make it a **learnable parameter**. This is called **PReLU** (Parametric ReLU):

$$
\text{PReLU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{if } x < 0 \end{cases}
$$

The formula is identical, but $\alpha$ is now learned during training via backpropagation, just like weights and biases. The network can discover the optimal slope for each layer (or even each neuron).

- If the optimal $\alpha$ is close to 0, PReLU learns to behave like ReLU
- If the optimal $\alpha$ is close to 1, the layer learns to be nearly linear
- In practice, learned $\alpha$ values often end up between 0.1 and 0.3

---

## Where Leaky ReLU Shows Up

- **GANs (Generative Adversarial Networks)**: Leaky ReLU is the standard activation for discriminator networks. The DCGAN paper recommended $\alpha = 0.2$ for the discriminator.
- **Object detection**: architectures like YOLO use Leaky ReLU
- **Any deep network worried about dead neurons**: if you are seeing many neurons with zero output after training, switching from ReLU to Leaky ReLU is the easiest fix
- **As a diagnostic**: if Leaky ReLU significantly outperforms ReLU on your task, it suggests that dead neurons were hurting performance