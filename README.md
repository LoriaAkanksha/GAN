# GAN
GANs are a type of generative model used in machine learning to produce new data samples that resemble a given training dataset. The key idea behind GANs is to train two neural networks simultaneously: a generator and a discriminator. The generator learns to produce synthetic data samples, while the discriminator learns to distinguish between real and fake samples.
### Generator:
The generator is a neural network component of a GAN that learns to generate synthetic data samples. It takes random noise vectors or other input data as input and produces fake data samples as output. The goal of the generator is to produce samples that are indistinguishable from real data samples. The generator is trained in opposition to the discriminator, with the objective of fooling the discriminator into classifying its outputs as real.
### Discriminator:
The discriminator is another neural network component of a GAN that learns to distinguish between real and fake data samples. It takes both real data samples from the training dataset and fake samples generated by the generator as input and outputs a probability score indicating the likelihood that each sample is real. The discriminator is trained to correctly classify real samples as real and fake samples as fake.
### Training Procedure:
During training, the generator and discriminator are trained alternately in a minimax game framework. The generator is trained to maximize the probability of the discriminator making a mistake (i.e., classifying fake samples as real), while the discriminator is trained to minimize its error in distinguishing between real and fake samples. This adversarial training process continues iteratively until the generator produces samples that are difficult for the discriminator to distinguish from real data.



### Tensors
Tensors are multi-dimensional arrays used to represent data. Tensors can have different ranks, which correspond to the number of dimensions. For example, a scalar (a single number) is a rank-0 tensor, a vector is a rank-1 tensor, a matrix is a rank-2 tensor, and so on. Tensors are fundamental data structures used for storing and manipulating data in neural networks.
### Batches
In training GANs and other machine learning models, data is typically divided into batches for efficient computation. A batch is a subset of the training data that is processed together during each training iteration. Batch size refers to the number of data samples in each batch. Training with batches allows for parallelization of computation and efficient use of resources such as GPU memory.


