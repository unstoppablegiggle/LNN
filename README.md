# LNN
This is a simple implementation of a Liquid Neural Network (LNN) using that classifies the MNIST data set from TensorFlow. The primary objective of this project was to give me a better
understanding of what is going on under the hood of neural networks. I chose an LNN in particular because of some novel features that allow for much smaller models
to perform comparable tasks. I first became interested in them whilst working with a Deep Protein Language Model that had the major and unfortunate limitation of
being very large and prohibitively time-consuming to use on a desktop PC. 
## Why are LNNs unique?
LNNs consist of three layers:

- Input layer (The data we put in)
- Liquid layer (This is the novel part)
- Output layer (What comes out)

What makes LNNs stand out compared to other Neural Nets is that time constants associated with their hidden states can vary. This means that instead of having to add a new node for a new state the LNN
can update an already existing node in response to the varying time inputs, allowing them to be much more compact and dynamic and better able to handle complex patterns efficiently. It makes them uniquely well suited
to tasks involving time series tasks such as natural language processing and tasks that involve real time adaptation such as autonomous vehicles

## Notes
Originally this LNN had an abysmal accuracy hovering around 31%, with some tweaking of hyperparameters this was improved to 91% by increasing the leak rate (how quickly the LNN forgets previous info and updates the new) to 0.75

## References
- Special thanks to [Gaudenz Boesch](https://viso.ai/deep-learning/what-are-liquid-neural-networks/) for his amazingly helpful work.
- Yann Lecunn for providing the MNIST database for all to learn from. [MNIST website](https://yann.lecun.com/exdb/mnist/)
