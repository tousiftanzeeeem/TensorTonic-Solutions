## What Is Entropy?

Entropy measures the **uncertainty** or **impurity** in a set of labels. In decision trees, we use entropy to quantify how mixed the classes are at each node.

**High entropy:** Labels are evenly mixed, maximum uncertainty

**Low entropy:** Labels are mostly one class, low uncertainty

**Zero entropy:** All labels are the same class, perfect purity

---

## The Entropy Formula

For a set with $C$ classes, where $p_i$ is the proportion of samples in class $i$:

$$
H = -\sum_{i=1}^{C} p_i \log_2(p_i)
$$

**Convention:** $0 \log_2(0) = 0$ (the limit as $p \to 0$)

The negative sign makes entropy positive (since $\log$ of fractions is negative).

---

## Understanding the Formula

Each term $-p_i \log_2(p_i)$ measures the "surprise" of seeing class $i$:

- Rare events ($p_i$ small) have high surprise ($-\log_2(p_i)$ is large)
- Common events ($p_i$ large) have low surprise
- We weight surprise by probability $p_i$

Entropy is the **expected surprise** when sampling from the distribution.

---

## Binary Classification Examples

For 2 classes with proportions $(p, 1-p)$:

$$
H = -p \log_2(p) - (1-p) \log_2(1-p)
$$

**Example 1: Perfect purity**

All samples class A: $p = 1.0$

$H = -1 \cdot \log_2(1) - 0 \cdot \log_2(0) = -1 \cdot 0 - 0 = 0$

Entropy is 0. No uncertainty.

**Example 2: Maximum impurity**

Half class A, half class B: $p = 0.5$

$H = -0.5 \cdot \log_2(0.5) - 0.5 \cdot \log_2(0.5)$

$= -0.5 \cdot (-1) - 0.5 \cdot (-1) = 0.5 + 0.5 = 1$

Entropy is 1 (maximum for binary). Maximum uncertainty.

**Example 3: Moderate impurity**

70% class A, 30% class B: $p = 0.7$

$H = -0.7 \cdot \log_2(0.7) - 0.3 \cdot \log_2(0.3)$

$= -0.7 \cdot (-0.515) - 0.3 \cdot (-1.737)$

$= 0.36 + 0.52 = 0.88$

---

## Multi-Class Example

**Node with 100 samples:**
- Class A: 50 samples ($p_A = 0.5$)
- Class B: 30 samples ($p_B = 0.3$)
- Class C: 20 samples ($p_C = 0.2$)

$$
H = -0.5 \log_2(0.5) - 0.3 \log_2(0.3) - 0.2 \log_2(0.2)
$$

$= -0.5 \cdot (-1) - 0.3 \cdot (-1.737) - 0.2 \cdot (-2.322)$

$= 0.5 + 0.521 + 0.464 = 1.485$

Maximum possible entropy for 3 classes is $\log_2(3) \approx 1.585$ (when all equal).

---

## Entropy Bounds

**Minimum entropy:** 0 (all samples same class)

**Maximum entropy:** $\log_2(C)$ (uniform distribution over $C$ classes)

- 2 classes: max = 1
- 3 classes: max = 1.585
- 4 classes: max = 2
- 10 classes: max = 3.322

---

## Entropy in Decision Trees

Decision trees split nodes to **reduce entropy**. The goal is to create child nodes that are purer than the parent.

**Information Gain** measures entropy reduction:

$$
\text{IG} = H(\text{parent}) - \sum_{\text{child}} \frac{n_{\text{child}}}{n_{\text{parent}}} H(\text{child})
$$

We choose the split that maximizes information gain (largest entropy reduction).

---

## Step-by-Step: Computing Node Entropy

**Given labels:** [A, A, B, A, B, B, A, C, A, C]

**Step 1: Count each class**
- A: 5
- B: 3
- C: 2
- Total: 10

**Step 2: Compute proportions**
- $p_A = 5/10 = 0.5$
- $p_B = 3/10 = 0.3$
- $p_C = 2/10 = 0.2$

**Step 3: Compute each term**
- $-0.5 \log_2(0.5) = -0.5 \times (-1) = 0.5$
- $-0.3 \log_2(0.3) = -0.3 \times (-1.737) = 0.521$
- $-0.2 \log_2(0.2) = -0.2 \times (-2.322) = 0.464$

**Step 4: Sum**
$H = 0.5 + 0.521 + 0.464 = 1.485$

---

## Handling Edge Cases

**All same class:**

Labels: [A, A, A, A]

$p_A = 1.0$, no other classes

$H = -1 \cdot \log_2(1) = -1 \cdot 0 = 0$

**Single sample:**

Labels: [B]

$p_B = 1.0$

$H = 0$ (certain, no entropy)

**Empty proportions:**

If a class has 0 samples, $p_i = 0$ and $0 \log_2(0) = 0$ by convention.

---

## Why Log Base 2?

Log base 2 gives entropy in **bits**:

- 1 bit of entropy = 1 yes/no question to determine the class
- 2 bits = 2 questions
- etc.

Other bases work too:
- Natural log ($\ln$): entropy in "nats"
- Log base 10: entropy in "dits"

The choice of base affects the scale but not the relative comparisons.

---

## Entropy vs. Gini Impurity

Both measure node impurity. Key differences:

**Entropy:**
- Uses logarithms
- Slightly higher computational cost
- More sensitive to changes in distribution
- Maximum value depends on number of classes

**Gini:**
- No logarithms, simpler formula
- Slightly faster to compute
- Ranges from 0 to 0.5 for binary
- Often produces similar trees

In practice, both work well. Gini is more common in implementations (e.g., scikit-learn default).

---

## The Entropy Curve (Binary Case)

For binary classification, entropy as a function of $p$:

- $H(0) = 0$
- $H(0.5) = 1$ (maximum)
- $H(1) = 0$

The curve is symmetric around $p = 0.5$ and concave (bulges upward).

This shape means:
- Any deviation from 50/50 reduces entropy
- The steepest reduction is near 50/50
- Near-pure nodes (p close to 0 or 1) have very low entropy

---

## Cross-Entropy Connection

Entropy is related to **cross-entropy loss** in neural networks:

$$
\text{Cross-Entropy} = -\sum_i y_i \log(\hat{p}_i)
$$

When $y$ is a one-hot label and $\hat{p}$ is the predicted distribution, minimizing cross-entropy pushes predictions toward the true label.

Shannon entropy is cross-entropy of a distribution with itself.