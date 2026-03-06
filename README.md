# Firsrt and Second order optimizations
a project in which i tried to implement Polyack's Heavy ball for first order and AdaHessian for second order

## First Order:
for the first order using Polyack's heavy ball, i added a dynamic learning rate, to improve convergence

## Second Order:
for the second order, I replaced the hessian with the Fisher information Matrix to decrease runtime and also added a dynamic
learning rate

## Results:
for both orders the optimal dynamic learning rate was cosine annealing.
The first order improved dramatically when starting with a high learning rate, gradually decreasing through cosine annealing.
The second order performed better but with a much higher runtime as expected.
I recommend using the second order out of rear that the optimal learning rate which i arrived at is an overfir
