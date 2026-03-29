# 🖼️ Foreground Extraction using ADMM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-Optimized-blueviolet.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Introduction
### Purpose of the Project
This project implements a robust mathematical framework to separate the background and foreground from a given image. Using **Robust Principal Component Analysis (RPCA)** optimized via the **Alternating Direction Method of Multipliers (ADMM)**, the algorithm intelligently decomposes an image into its constituent components.

### The Problem Being Solved
Traditional Principal Component Analysis (PCA) is highly sensitive to gross errors and outliers. In the context of computer vision, a moving foreground object behaves as an unpredictable, sparse outlier that corrupts the low-rank structure of the static background. This project solves this by employing $\ell_1$-norm regularization for the outliers, allowing for exact recovery of the underlying low-rank background matrix and the sparse foreground matrix, even in the presence of severe corruption.

---

## 🔬 Project Overview
At its core, the problem is formulated as a matrix decomposition task.

### Mathematical Formulation & Derivations

Let $M$ be our observed image matrix. We aim to decompose $M$ into two distinct matrices:

$$
M = L + S
$$

Where:
* **$L$**: The low-rank background matrix (representing the static or slowly changing background).
* **$S$**: The sparse foreground matrix (representing moving objects or distinct foreground elements).

#### Optimization Objective
To achieve this decomposition, we minimize the nuclear norm of $L$ (to encourage low rank) and the $\ell_1$-norm of $S$ (to encourage sparsity), subject to the data constraint:

$$
\min_{L, S} \lVert L \rVert_{*} + \lambda \lVert S \rVert_{1} \quad \text{subject to} \quad M - L - S = 0
$$

*   **$\lVert L \rVert_{*}$**: The nuclear norm (sum of singular values) of $L$.
*   **$\lVert S \rVert_{1}$**: The $\ell_1$-norm (sum of absolute values) of $S$.
*   **$\lambda$**: A tuning parameter that balances the rank of $L$ against the sparsity of $S$.

#### Augmented Lagrangian
To solve this constrained optimization problem, we use the Augmented Lagrangian method. Let $R = L + S - M$, and let $Y$ be the dual variable matrix. The Augmented Lagrangian $\mathcal{L}(L, S, Y)$ is defined as:

$$
\mathcal{L}(L, S, Y) = \lVert L \rVert_{*} + \lambda \lVert S \rVert_{1} + \langle Y, R \rangle + \frac{\mu}{2} \lVert R \rVert_{F}^{2}
$$

*   **$\langle Y, R \rangle$**: The inner product of the dual variable and the residual, which enforces the constraint.
*   **$\frac{\mu}{2} \lVert R \rVert_{F}^{2}$**: The penalty term based on the Frobenius norm squared, which ensures strict convexity and improves convergence. $\mu$ is the penalty parameter.

Using ADMM, we minimize $\mathcal{L}$ by updating $L$, $S$, and $Y$ alternately.

#### 1. Update Equation for $L$
We isolate $L$ and minimize the Lagrangian with respect to it. Let:

$$
Q = M - S^{(k)} + \frac{1}{\mu} Y^{(k)}
$$

Then, the localized optimization problem for $L$ becomes:

$$
L^{(k+1)} = \arg\min_{L} \lVert L \rVert_{*} + \frac{\mu}{2} \lVert L - Q \rVert_{F}^{2}
$$

**Solution:** This is solved using the **Singular Value Thresholding (SVT)** operator. If $Q = U \Sigma V^T$ is the Singular Value Decomposition (SVD) of $Q$, then:

$$
L^{(k+1)} = U \ \text{diag}\left( \max\left( \sigma_i - \frac{1}{\mu}, 0 \right) \right) V^T = \text{SVT}_{1/\mu}(Q)
$$

*Explanation:* We perform SVD on $Q$, shrink the singular values by $1/\mu$, and reconstruct the matrix. This eliminates small singular values, enforcing a low-rank structure.

#### 2. Update Equation for $S$
Next, we isolate $S$. Let:

$$
A = M - L^{(k+1)} + \frac{1}{\mu} Y^{(k)}
$$

The localized optimization problem for $S$ is:

$$
S^{(k+1)} = \arg\min_{S} \lambda \lVert S \rVert_{1} + \frac{\mu}{2} \lVert S - A \rVert_{F}^{2}
$$

**Solution:** This is solved via the **Element-wise Soft-Thresholding** operator:

$$
S^{(k+1)} = \text{Soft}_{\lambda/\mu}(A)
$$

Where the Soft-Thresholding function operates element-wise:

$$
\text{Soft}_{\tau}(x) = \begin{cases} 
x - \tau, & x > \tau \\ 
x + \tau, & x < -\tau \\ 
0, & \text{otherwise} 
\end{cases}
$$

*Explanation:* This operator shrinks all pixel values of $A$ towards zero by $\lambda/\mu$. Pixels with small intensities (mostly background noise left in $S$) are forced to exactly zero, ensuring $S$ is mathematically sparse.

#### 3. Update Dual Variable $Y$
Finally, we perform gradient ascent on the dual variable $Y$:

$$
Y^{(k+1)} = Y^{(k)} + \mu (M - L^{(k+1)} - S^{(k+1)})
$$

*Explanation:* This adjusts the dual variable based on the residual error. As the algorithm converges, $M - L - S$ approaches zero, and $Y$ stabilizes.

### Key Concepts Used
*   **Low-Rank Decomposition:** Modeling static background mathematically as a matrix with minimal linearly independent dimensions.
*   **Sparse Modeling:** Identifying foreground objects as sparse, localized anomalies in the matrix.
*   **ADMM (Alternating Direction Method of Multipliers):** A powerful algorithmic framework that breaks a massive optimization problem into smaller, easily solvable sub-problems.
*   **SVD (Singular Value Decomposition):** Used iteratively to compute and enforce the nuclear norm mathematically.

---

## ⚙️ Methodology / Approach
The implementation follows an iterative block-coordinate descent approach via ADMM:
1.  **Initialization:** The algorithm starts by initializing $L$, $S$, and $Y$ as zero matrices. The $\lambda$ parameter is dynamically set as $1 / \sqrt{\max(m, n)}$. Parameter $\mu$ is initialized based on the matrix's $\ell_2$-norm.
2.  **Alternating Optimization:** In a loop of `max_iter`, the algorithm sequentially computes the SVT to update the low-rank background ($L$) and applies soft-thresholding to update the sparse foreground ($S$).
3.  **Dual Accumulation:** The dual variable $Y$ and penalty term $\mu$ are updated dynamically to speed up convergence.
4.  **Convergence Check:** The loop breaks early if the reconstruction error $\frac{\lVert M - L - S \rVert_{F}}{\lVert M \rVert_{F}}$ drops below a defined tolerance (`tol = 1e-7`), ensuring computational efficiency.
5.  **Post-processing:** The extracted matrices $L$ and $S$ are bounded back into 8-bit $[0, 255]$ space and thresholded for clean visualization.

---

## 📊 Results
The algorithm takes an ordinary, single-channel (grayscale) image normalized to $[0, 1]$ and strictly isolates the foreground layer from the background layer. 
*   **Background Output ($L$):** The output matrix retains only the structural, low-frequency data. 
*   **Foreground Output ($S$):** Moving or distinct objects appear as high-contrast patches against a strictly zero (black) background. Applying binary thresholding cleans this output to provide a distinct mask of the foreground mapping.
<img width="327" height="377" alt="Image" src="https://github.com/user-attachments/assets/1f028793-f19e-43cc-bd90-21c2a3840769" />
<img width="314" height="389" alt="b47fb838-3cab-4059-a4ef-1cbfdc76a6fd" src="https://github.com/user-attachments/assets/4c4b3830-c2ef-47f1-b2c3-e4e5d30bc279" />


---

## 💡 Conclusion
### Summary
This project successfully establishes a mathematical optimization routine capable of decomposing an image into low-rank and sparse matrices. By applying Robust PCA evaluated through ADMM, we demonstrate precise foreground-background separation, which acts as a foundational technique in video surveillance, target tracking, and computer vision.

### Limitations & Improvements
*   **Computational Bottleneck:** The primary limitation is the $\mathcal{O}(n^3)$ cost of performing Singular Value Decomposition (SVD) at every iteration, which scales poorly for very large or high-resolution images.
*   **Possible Improvements:**
    *   Transitioning to **Randomized SVD** or **Truncated SVD** to drastically cut computation time.
    *   Implementing a tensor-based RPCA approach directly mapped to RGB images instead of collapsing frames into grayscale.
    *   Accelerating the matrix calculations leveraging GPU computation (e.g., CuPy or PyTorch). 

---

## 📚 References
*   [Foreground-Background Separation via Generalized Nuclear Norm and Structured Sparse Norm Based Low-Rank and Sparse Decomposition](https://ieeexplore.ieee.org/document/9085351) - IEEE Xplore, 2020.
