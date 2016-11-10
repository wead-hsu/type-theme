---
layout: post
title: Linear Algebra and Its Application - A Summary
sitetype: blog
categories: mathematics
---

此文仅为经典线性代数教材《Linear Algebra and Its Application》的学习记录，包括一些数学概念，定义，个人理解的摘要。




### 第一节：Row Reduction and Echelon Forms

Echelon form: 行消元后的矩阵
Reduced echelon form: 行消元并且leading entry为1的矩阵。
Echelon form and reduced echelon form are row equivalent to the original form.
$$Span\{v_1, v_2, v_3,...... v_p\}$$ is the collection of all vectors that can be written in the form $$ c_1*v_1 + c_2*v_2 + ...... c_p*v_p $$ with $$c_1,....c_p$$scalars.
Ax = 0 has a nontrival solution if and only if the equation has at least one free variable.(not full column rank)
Ax = b 的解等于 Ax = 0 和 特解的和。
解线性方程组流程P54。
线性无关指任何向量不能组合成其中一个向量。
Ax = b : ColA1 * x1 + ColA2 * x2 +.... ColAm * xm = b
Matrix Transformations: T(x) = Ax is linear transformation.
转换矩阵是各维单位转换后的组合。A = [T(e1) T(e2) .. T(en)]
A mapping T: R^n -> R^m is said to be onto R^m if each b in R^m is the image of at least one x in R^n. (Ax = b 有解)
A mapping T: R^n -> R^m is said to be one-to-one R^m if each b in R^m is the image of at most one x in R^n.

### 第二节：Matrix Operation

Each column of AB is a linear combination of the columns of A using weightings from the corresponding columns of B. AB = A[b1  b2 b3 b4 ,,, bp] = [Ab1 Ab2 ... Abp]
Each row of AB is a linear combination of the columns of B using weightings from the corresponding rows of A.
Warning: AB != BA. AB = AC !=> B = C. AB = 0 !=> A = 0 or B = 0
逆矩阵的定义：A-1*A = A*A-1 = E. 可以推导出A为方阵，详见Exercise 23-25 ，Section 2.1. A可逆的充要条件为A满秩（行列式不等于0）。
对[A I] 做行消元可以得到[I A-1]
矩阵满秩的所有等价定义:P129,P179.
LU分解：A = LU，其中L为对角元素为1,的下半方阵，U为m*n的上半矩阵。L为变换矩阵的乘机的逆，U为A的Echelon form。计算L不需要计算各变换矩阵。详见P146。
subspace, column space, null space的定义。
A = m*n => rank(A) + rank(Nul(A)) = n.
The dimension of a nonzero subspace H, denoted by dim H, is the numbers of vectors in any basis for H. The dimension of the zero subspace {0} us defined to be zero.

### 第三节：Introduction to Determinants

determinant的定义和计算方式。
行消元不改变行列式值。交换行改变正负号。某一行乘以k，那么行列式乘以k。
三角矩阵的行列式为对角元素的乘积。
 $$det(AB) = det(A) * det(B)$$。
Let A be an invertible n\*n matrix. For any b in R^n, the unique solutionx of Ax = b has entries given by xi = det Ai(b)/det(A)。 Ai(b) 表示用b替换A的第i行。
由5可以推导出A^-1 = 1/det(A) * adj A. adj A = [(-1)^i+j* det(Aji)]
行列式与体积的关系：平行几何体的面积或者体积等于|det(A)|。而且 det(Ap) = det(A)*det(p)

### 第四节：Vector Spaces

An indexed set {v1, v2, ... ... vp} of two or more vectors, with vi != 0, is linearly dependent, if and only if some vj (with j > 1) is a linear combination of the preceding vectors.
Elementary row operation  on a matrix do not affect the linear dependence relations among the columns of the matrix.
Row operations can change the column space of a matrix.
x = Pb [x]b: we call Pb the change-of-coordinates matrix from B to the standard basis in R^n.
Let B and C be bases of a vector space V. Then there is a unique n*n matrix P_C<-B such that [x]c = P_C<-B [x]b. The columns of P_C<-B are the C-coordinate vectors of the vectors in the basis B, that is P_C<-B = [[b1]c [b2]c ... [bn]c]. [ C B ] ~ [ I P_C<-B]

### 第五节：Eigenvectors and Eigenvalues

$$Ax =\lambda * x$$
不同特征值对应的特征向量线性无关。
$$det(A - λ*I) = 0$$. 因为$$(A - λ*I)$$有非零解。
A is similar to B if there is an invertible matrix P such that P^-1AP = B. They have same eigenvalues.
矩阵能够对角化的条件是有n个线性无关的特征向量（特征向量有无穷多个，线性无关向量的数量最多为n）。
特征空间的维度小于等于特征根的幂。当特征空间的维度等于特征根的幂，矩阵能够对角化。
相同坐标变换矩阵在不同维度空间坐标系下的转换：P328。相同坐标变换矩阵在不同坐标系的转换：P329。其实都是一样的。
Suppose A = PDP^-1, where D is a diagonal n*n matrix. If B is the basis for R^n formed from the columns of P, then D is the B-matrix for the transformation x ->Ax. 当坐标系转换为P时，转换矩阵对应变成对角矩阵。
复数系统。
迭代求特征值和特征向量。 先估计一个特近的特征值和一个向量$$x_0$$（其中的最大元素为1）。然后迭代，迭代流程详见P365。迭代可以得到最大特征值的原因如下：因为$$(\lambda_1)^{-k}A^kx\rightarrow c_1v_1$$,所以对于任意$$x$$,当k趋近无穷的时候，$$A^kx$$会和特征向量同向。虽然$$\lambda$$和$$c_1v_1$$都未知，但是由于$$Ax_k$$会趋近$$\lambda*x_k$$,我们只要令$$x_k$$的最大元素为1，就能得到$$\lambda$$。

### 第六节 ：Inner Product, Length, and Orthogonality

$$(Row A)^{\bot} = Nul A$$ and $$(Col A)^{\bot} = Nul A^{\top}$$. 这很显然，其中$$A^{\bot}$$表示与A空间垂直的空间。
An orthogonal basis for a subspace W of $$R^n$$ is a basis for W that is also an orthogonal set.
一个向量在某一维的投影：$$\hat{y} = proj_L y = \frac{y\cdot u}{u\cdot u}u$$.
An set is an orthonormal set if it is an orthogonal set of unit vectors.
An m*n matrix U has orthonormal columns if and only if $$U^\top U = I$$
一个向量在某一空间的投影：$$\hat{y} = proj_w y = \frac{y\cdot u_1}{u_1\cdot u_1}u_1 + \frac{y\cdot u_2}{u_1\cdot u_2}u_2 + ... + \frac{y{\cdot}u_p}{u_p\cdot u_p}u_p.$$
如何将一堆向量弄成正交单位向量: repeat 3.
QR分解：如果A有线性无关的列向量，那么可以分解成Q（正交向量）和R（上三角矩阵，就是原坐标在正交坐标系的系数）$$Q^{\top}A=Q^{\top}(QR) = IR = R$$
最小平方lse（机器学习基础：非贝叶斯条件下的线性拟合问题），由$$A^{\top}(b-A\hat{x})=0$$得到$$\hat{x}=(A^\top A)^{-1}A^{\top}b$$。如果A可逆，此式可以化简。如果可以做QR分解，那么$$\hat{x}=R^{-1}Q^{\top}b$$.
函数内积的概念。

### 第七节：Diagonaliztion of Symmetric matrixs

如果一个矩阵是对称的，那么它的任何两个特征值所对应的特征空间是正交的。
矩阵可正交对角化等价于它是一个对称矩阵。
$$A=PDP^{-1}$$可以得到PCA（机器学习算法主成分分析，对协方差矩阵（对称）做对角化）
将二次方程转化成没有叉乘项的形式。x=Py,  $$A = PDP^{-1}$$.
对于二次函数$$x^{\top}Ax$$，在|x| = 1的条件下，最大值为最大特征值，最小值为最小特征值。如果最大特征值（$$x^{\top}u_1$$）不能选，则选择次之。
正交矩阵P大概意思就是在该坐标系下，函数比较对称，D为坐标轴的伸展比例。
SVD分解（该书的最后一个内容，蕴含了很多上述的内容）是要将矩阵分解成类似PDP^-1的形式，但是不是任何矩阵都能表示成这种形式（有n个线性无关的特征向量，正交的话还要是对称矩阵）。其中$$A=U{\Sigma}V^{\top}$$，$${\Sigma}$$是A的singular value（$$A^{\top}A$$的特征值的开方），V是$$A^{\top}A$$的对应特征向量，U是$$AV$$的归一化。AV内的向量是垂直的。$$U{\Sigma}$$是AV的另外一种表示。
