# Deep Learning Optimization

A repository for training recurrent neural networks (RNNs) using either: (i) backpropagation, or (ii) penalized backpropagation.

### File Descriptions

- `src/adjoint_regularizers.py` 
	- adjoint penalty functions, $G(\{\lambda_t\})$
- `src/architectures.py` 
	- class implementations for a variety of recurrent neural network architectures
- `src/coadjoint.py` 
	- class implementation of penalized backpropagation model
- `src/dependencies.py`
	- additional helper functions
	

### General Problem Set Up

**Definitition (Recurrent Neural Network):** For inputs $\\{u_1,\dots,u_T\\}\subset\mathbb{R}^p$, arbitrary initial state $x_0\in\mathbb{R}^d$, an activation function $\sigma: \mathbb{R}^d\rightarrow\mathbb{R}^d$, and an output function $\phi: \mathbb{R}^l\rightarrow\mathbb{R}^l$, a *Recurrent Neural Network* (RNN) is defined by
```math
\begin{equation}
	\begin{aligned}
		x_i =& \sigma(Wx_{i-1} + Ru_i + b_1), \quad i=1,\dots,T \\
		\hat{y}=& \phi(Vx_T + b_o),
	\end{aligned}
\end{equation}
```
where $W\in\mathbb{R}^{d\times d}$, $R\in\mathbb{R}^{d\times p}$, $V\in\mathbb{R}^{l\times d}$ and $b_o\in\mathbb{R}^l$.


#### [Backpropagation] Empirical Risk Minimization Problem

For a set of examples, $\\{(u_1^j, u_2^j,\dots,u_T^j, y^j)\ :\  j=1,\dots,N\ \\}$, the ERM training problem is to solve:

```math
\begin{equation}
	\begin{aligned}
		\min_{W,R,b_1,V,b_o}\quad &\sum_{j=1}^{N}L(\hat{y}^j,y^j) \\
		s.t.\quad  &x_i = \sigma(Wx_{i-1} + Ru_i + b_1), \quad i=1,\dots,T \\
		\quad &\hat{y}= \phi(Vx_T + b_o),
	\end{aligned}
\end{equation}
```


#### [Penalized Backpropagation] Regularized Empirical Risk Minimization Problem 

For a set of examples, $\\{(u_1^j, u_2^j,\dots,u_T^j, y^j)\ :\  j=1,\dots,N\ \\}$, the regularized ERM training problem is to solve:

```math
\begin{equation}
	\begin{aligned}
		\min_{W,R,b_1,V,b_o}\quad &\sum_{j=1}^{N}L(\hat{y}^j,y^j) + G(\{\lambda_t\}) \\
		s.t.\quad  &x_i = \sigma(Wx_{i-1} + Ru_i + b_1), \quad i=1,\dots,T \\
		\quad &\hat{y}= \phi(Vx_T + b_o) \\
		\quad &\lambda_i = W'\sigma^{(1)}(Wx_i + Ru_{i+1} + b_1)'\lambda_{i+1}, \quad i=1,\dots,T \\
		\quad &\lambda_T = -V'\phi^{(1)}(Vx_T+b_o)'\frac{\partial L}{\partial\hat{y}} \\
	\end{aligned}
\end{equation}
```
where $\phi^{(1)}$ and $\sigma^{(1)}$ (applied component-wise) represent the derivatives of $\phi$ and $\sigma$, respectively; and $G(\{\lambda_t\})$ is a twice differentiable function of the (backward) adjoint states. For example,
```math
\begin{equation}
	G(\{\lambda_t\}) = \frac{1}{T}\sum_{i=1}^{T}||\lambda_i||_2^2 - \bigg{(}\frac{1}{T}\sum_{j=1}^{T}||\lambda_j||_2\bigg{)}^2
\end{equation}
```