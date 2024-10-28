# Deep Learning Optimization

A repository for training recurrent neural networks (RNNs) using either: (i) backpropagation, or (ii) penalized backpropagation.

### Problem Set Up

**Definitition (Recurrent Neural Network):** For inputs $\\{u_1,\dots,u_T\\}\subset\mathbb{R}^p$, arbitrary initial state $x_0\in\mathbb{R}^d$, an activation function $\sigma: \mathbb{R}^d\rightarrow\mathbb{R}^d$, and an output function $\phi: \mathbb{R}^l\rightarrow\mathbb{R}^l$, a \emph{Recurrent Neural Network} (RNN) is defined by
```math
\begin{equation}
	\begin{aligned}
		x_i =& \sigma(Wx_{i-1} + Ru_i + b_1), \quad i=1,\dots,N \\
		\hat{y}=& \phi(Vx_N + b_o),
	\end{aligned}
\end{equation}
```
where $W\in\mathbb{R}^{d\times d}$, $R\in\mathbb{R}^{d\times p}$, $V\in\mathbb{R}^{l\times d}$ and $b_o\in\mathbb{R}^l$.


### Backpropagation Optimization




### Penalized Backpropagation Optimization