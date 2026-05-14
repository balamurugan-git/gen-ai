# Deep Learning — Class Notes

---

## 1. Introduction & History

Deep Learning as a field traces back to early neural network research in 1958 when Frank Rosenblatt introduced the Perceptron. The core philosophy is to build computational models that mimic the way the human brain processes information — through layers of interconnected neurons. Over decades, research went through cycles of excitement and stagnation ("AI winters") before GPU power and big data reignited the field around 2010. Today, DL powers image recognition, NLP, speech, and more.

---

## 2. Why Deep Learning Now?

Three forces converged to make Deep Learning practical: (1) **Data** — technologies like Hadoop, Spark, and cloud storage made it possible to collect and process massive datasets that neural networks need to learn from. (2) **Hardware** — GPUs (originally built for gaming) turned out to be ideal for the parallel matrix operations in neural networks, reducing training time from weeks to hours. (3) **Algorithms** — improvements in activation functions, optimizers, and regularization techniques resolved many training problems from earlier decades.

---

## 3. Neural Network Basics — Perceptron

The Perceptron is the simplest neural network — it has a single layer of neurons. It takes input features (like glucose level, age, BMI in a diabetes dataset), multiplies each by a weight, sums them, and passes through an activation function to produce output (diabetic: 1 or non-diabetic: 0). It is a **binary classifier** — it can only separate data into two classes. While limited on its own, the Perceptron is the foundational building block from which deep networks are constructed.

```
┌─────────────────────────────────────────────────────────────┐
│                    PERCEPTRON STRUCTURE                      │
│                                                             │
│   Input Layer        Hidden Layer       Output Layer        │
│                                                             │
│   x1 ──w1──►                                               │
│   x2 ──w2──► [Σ xiWi + bias] ──► Activation ──► ŷ (0/1)  │
│   x3 ──w3──►                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. How the Brain Analogy Works

When a human sees a car, light enters the eyes and electrical signals travel through millions of neurons. Each neuron fires (activates) or doesn't based on patterns it has previously learned. A newborn baby has no concept of a "car" — it learns only through repeated exposure and labeling by adults. Neural networks are the same: they start with random weights and have no knowledge, but through training on labeled data (input features + correct output labels), they gradually learn the patterns that distinguish one class from another.

---

## 5. Forward Propagation

Forward propagation is the process of feeding input data through the network from left (input layer) to right (output layer) to produce a prediction. At each neuron: (1) all incoming inputs are multiplied by their respective weights and summed — `y = Σ xᵢwᵢ + bias`; (2) a **bias** term is added to ensure the neuron can still fire even when all inputs are zero; (3) the result is passed through an **activation function** which decides how strongly the neuron activates. The final output is the network's prediction `ŷ`.

```
┌────────────────────────────────────────────────────────────────────┐
│                    FORWARD PROPAGATION FLOW                        │
│                                                                    │
│  Step 1           Step 2              Step 3         Step 4        │
│                                                                    │
│  Input          Weighted Sum          Activation     Output        │
│  Features  ──►  Σ(xᵢwᵢ) + bias  ──►  f(y)      ──►  ŷ            │
│  (x1,x2,x3)     = w᷊x               sigmoid/relu   (0 or 1)       │
│                                                                    │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │ x1 ──w1──┐                                                │    │
│  │ x2 ──w2──┼──► [Σ + b] ──► Act(y) ──w4──► [Act] ──► 0/1 │    │
│  │ x3 ──w3──┘       ↑                                        │    │
│  │                 Bias                                       │    │
│  └───────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```


---

## 6. Activation Functions

An activation function introduces **non-linearity** into the network — without it, stacking multiple layers would be mathematically equivalent to a single linear layer. The analogy of a hot object is perfect: your hand's reflex neuron "decides" to trigger only if the heat exceeds a threshold. **Sigmoid** squashes any value into (0, 1) and is ideal for binary output layers. **ReLU** outputs 0 for negatives and the input itself for positives — it trains faster and avoids vanishing gradients. **Leaky ReLU** fixes the "dying ReLU" problem by allowing a small slope for negatives. **Tanh** outputs between (-1, 1) and is zero-centered, helping in hidden layers.

```
┌────────────────────────────────────────────────────────────────────┐
│                   ACTIVATION FUNCTIONS COMPARISON                  │
│                                                                    │
│  SIGMOID              RELU               TANH                      │
│  f(x) = 1/(1+e⁻ˣ)    f(x) = max(0,x)   f(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)│
│  Range: (0, 1)        Range: [0, ∞)     Range: (-1, 1)            │
│                                                                    │
│    1 ┤  ──────         ┤  /              1 ┤   ───────             │
│      │ /               │ /               0 ┤ ─/                   │
│    0 ┤─                ┤──────          -1 ┤─────                  │
│                                                                    │
│  USE: Binary Output   USE: Hidden Layers  USE: Hidden Layers       │
│  CON: Vanishing grad  CON: Dying ReLU     CON: Vanishing grad      │
│                                                                    │
│  LEAKY RELU: f(x) = max(0.01x, x) → fixes dying ReLU problem      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 7. Loss Function

The loss function is the **report card** of the neural network — it measures how far the network's predictions are from the true labels. For binary classification, **Binary Cross-Entropy** is used; for regression, **Mean Squared Error (MSE)** = ½(y - ŷ)² is common. The goal during training is to **minimize** the loss. A high loss means the model is making bad predictions; a loss close to zero means predictions match the actual values well. Every training loop works toward reducing this number.

```
┌────────────────────────────────────────────────────────────────────┐
│                        LOSS FUNCTION CONCEPT                       │
│                                                                    │
│  Ground Truth (y) ──────────────────────┐                         │
│                                          ▼                         │
│                               Loss = f(y, ŷ)  ◄── Predicted (ŷ)  │
│                                          │                         │
│                              ┌───────────┴───────────┐            │
│                              │                       │            │
│                         Loss ≈ 0               Loss >> 0          │
│                    (Good Prediction)        (Bad Prediction)       │
│                              │                       │            │
│                         ✅ Done                ❌ Backpropagate   │
│                                                                    │
│  MSE Loss (Regression):   L = ½(y - ŷ)²                          │
│  Binary Cross-Entropy:    L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]       │
└────────────────────────────────────────────────────────────────────┘
```

---

## 8. Backward Propagation

Backpropagation is the learning mechanism of the network. When the loss is large, the network needs to figure out **which weights contributed to the error** and by how much. It does this by flowing the error **backwards** from output to input, computing the gradient (partial derivative) of the loss with respect to each weight. The **chain rule of calculus** is used to efficiently compute these gradients layer by layer. This is how deep networks with millions of parameters can be trained — each weight gets a precise signal telling it which direction to adjust.

```
┌────────────────────────────────────────────────────────────────────┐
│                    BACKPROPAGATION FLOW                            │
│                                                                    │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐        │
│  │  Input  │    │Hidden 1 │    │Hidden 2 │    │ Output  │        │
│  │  Layer  │    │  Layer  │    │  Layer  │    │  Layer  │        │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘        │
│       │              │              │              │              │
│       ◄──── ∂L/∂w1 ──◄──── ∂L/∂w2 ──◄──── ∂L/∂w3 ──◄── Loss     │
│                                                                    │
│  Chain Rule:  ∂L/∂w1 = (∂L/∂o₂) × (∂o₂/∂o₁) × (∂o₁/∂w1)        │
│                                                                    │
│  Weight Update: W_new = W_old − η × (∂L/∂W_old)                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 9. Optimizers & Gradient Descent

An optimizer is the algorithm that **executes** the weight updates calculated by backprop. **Gradient Descent** is the foundational optimizer — it moves weights in the direction of the **negative gradient** (steepest descent) of the loss function, like a ball rolling downhill toward the lowest valley (global minima). The **learning rate** (η ≈ 0.01) controls the step size: too large and you overshoot the minimum; too small and training is painfully slow. Advanced optimizers like **Adam**, **RMSprop**, and **SGD with Momentum** build on gradient descent with adaptive learning rates and momentum terms to converge faster and more reliably.

```
┌────────────────────────────────────────────────────────────────────┐
│                    GRADIENT DESCENT VISUALISED                     │
│                                                                    │
│  Loss                                                              │
│    │     ●  ← Start (high loss)                                    │
│    │      \                                                        │
│    │       ●  ← Step with η                                        │
│    │        \                                                      │
│    │         ●                                                     │
│    │          \                                                    │
│    │           ★  ← Global Minima (minimum loss)                  │
│    └──────────────────────────── Weights                          │
│                                                                    │
│  Formula:   W_new = W_old − η × (∂Loss / ∂W_old)                 │
│                                                                    │
│  η too large → Overshoots minimum (diverges)                      │
│  η too small → Very slow convergence                              │
│  η ≈ 0.01   → Balanced convergence  ✅                            │
└────────────────────────────────────────────────────────────────────┘
```

---

## 10. Full Training Loop — Summary

The complete training loop of a neural network is a cycle: data enters through input layers, flows forward through weighted connections with bias and activation functions (forward propagation), a loss is computed comparing prediction vs truth, and then the optimizer backpropagates the error to update every weight in the network. This loop runs for many **epochs** (full passes through the training data) until the loss converges to a minimum. Each iteration makes the network slightly smarter — this is the essence of machine learning.

```
┌────────────────────────────────────────────────────────────────────┐
│                  COMPLETE NEURAL NETWORK TRAINING LOOP             │
│                                                                    │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │               FORWARD PROPAGATION                       │     │
│   │                                                         │     │
│   │  ① Input (x1, x2, x3)                                  │     │
│   │       │                                                 │     │
│   │       ▼                                                 │     │
│   │  ② Multiply by Weights (w1, w2, w3)                    │     │
│   │       │                                                 │     │
│   │       ▼                                                 │     │
│   │  ③ Add Bias  →  Σ(xᵢwᵢ) + b                           │     │
│   │       │                                                 │     │
│   │       ▼                                                 │     │
│   │  ④ Activation Function  →  Act(y)  →  ŷ               │     │
│   └─────────────────────────────────────────────────────────┘     │
│                          │                                         │
│                          ▼                                         │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │               BACKWARD PROPAGATION                      │     │
│   │                                                         │     │
│   │  ⑤ Loss Function  →  L = f(y, ŷ)  [Is loss high?]     │     │
│   │       │ YES — update weights                            │     │
│   │       ▼                                                 │     │
│   │  ⑥ Optimizer computes gradients (chain rule)           │     │
│   │       │                                                 │     │
│   │       ▼                                                 │     │
│   │  ⑦ Update Weights:  W_new = W_old − η·(∂L/∂W_old)     │     │
│   │       │                                                 │     │
│   │       └──────────────────────────────────► Repeat      │     │
│   └─────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 11. Multi-Layered Neural Network

A Multi-Layer Perceptron (MLP) or Deep Neural Network stacks multiple hidden layers between the input and output. Each layer learns progressively **abstract representations** — early layers learn simple features (edges in images, common word patterns in text), while deeper layers combine those into complex concepts (faces, sentence meaning). The more hidden layers, the "deeper" the network — this is why the field is called Deep Learning. However, deeper networks are harder to train (vanishing gradients, overfitting) and require careful architectural choices, regularization, and sufficient data.



```
┌────────────────────────────────────────────────────────────────────┐
│                   MULTI-LAYERED NEURAL NETWORK                     │
│                                                                    │
│  Input         Hidden Layer 1    Hidden Layer 2      Output        │
│  Layer              (HL1)            (HL2)           Layer         │
│                                                                    │
│  x1 ──○──────────○──────────────○                                  │
│         \       / \             / \                                 │
│  x2 ──○──\──── /   \───────── /   \──────○──── ŷ                  │
│           \  ○       ○───────○     ○                               │
│  x3 ──○────○   \   /   \   /   \ /                                 │
│               \ ○       ○       ○                                  │
│                ○   \   /   \   /                                   │
│                     ○       ○                                      │
│                                                                    │
│  Each arrow = a weight (learned during training)                   │
│  Each ○     = a neuron (applies activation function)               │
│  Depth      = number of hidden layers                              │
└────────────────────────────────────────────────────────────────────┘
```

---

## 12. Reference: Activation Functions, Loss Functions & Optimizers

| Category | Type | Use Case |
|---|---|---|
| **Activation** | Sigmoid | Binary classification output |
| **Activation** | ReLU | Hidden layers (default choice) |
| **Activation** | Leaky ReLU | When neurons are "dying" |
| **Activation** | Tanh | Hidden layers, zero-centered |
| **Activation** | Softmax | Multi-class output layer |
| **Loss** | Binary Cross-Entropy | Binary classification |
| **Loss** | Categorical Cross-Entropy | Multi-class classification |
| **Loss** | MSE | Regression tasks |
| **Optimizer** | SGD | Simple baseline |
| **Optimizer** | Adam | General purpose, adaptive |
| **Optimizer** | RMSprop | Recurrent networks |
| **Optimizer** | AdaGrad | Sparse data |

---

*Notes enhanced for classroom delivery — original diagrams preserved alongside structured flowcharts.*
