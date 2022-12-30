# Max Entropy Solver for the game of Wordle

In the game Wordle, the objective is to find the hidden word by guessing different letter combinations and receiving feedback on which letters are correct and in the correct position (a "hit") and which letters are correct but in the wrong position (a "pseudo-hit").

Using the maximum entropy solver, we can approach this problem by defining a set of possible solutions (i.e., different letter combinations) and selecting the solution that maximizes the entropy while satisfying the constraints imposed by the hits and pseudo-hits. For example, if the hidden word is "apple" and the player has received the following feedback: 2 hits and 1 pseudo-hit, we can define the set of possible solutions as all letter combinations that have 2 a's and 1 p in the correct positions, and select the solution with the highest entropy.

One advantage of using the maximum entropy solver in this context is that it can handle situations where there is incomplete or uncertain information about the hidden word. By maximizing the entropy of the solution, the maximum entropy solver can take into account the uncertainty in the data and find the most likely solution that is consistent with the available information.

Another advantage of the maximum entropy solver is that it is simple to implement and can be applied to a wide range of problems. However, it is important to note that the maximum entropy solver may not always find the optimal solution to the Wordle game, and may be sensitive to the choice of constraints or observations used to define the solution space.

## Usage

The solver is implemented in Solver.py , can be modified to for use online or used inside main.py where wordle has been implemented with pygame. the predictions are in the console.

## Requirements

- Python >= 3.6
- pygame
- numpy

## References
[1] C. E. Shannon, “A Mathematical Theory of Communication,” Bell Syst. Tech. J., vol. 27, no.
3, pp. 379–423, 1948, doi: 10.1002/j.1538-7305.1948.tb01338.x.
[2] O. Maroney, “Information Processing and Thermodynamic Entropy,” in The Stanford 
Encyclopedia of Philosophy, Fall 2009., E. N. Zalta, Ed. Metaphysics Research Lab, Stanford 
University, 2009. Accessed: Apr. 12, 2022. [Online]. Available: 
https://plato.stanford.edu/archives/fall2009/entries/information-entropy/
[3] R. V. L. Hartley, “Transmission of Information1,” Bell Syst. Tech. J., vol. 7, no. 3, pp. 535–
563, 1928, doi: 10.1002/j.1538-7305.1928.tb01236.x.
[4] K. M and M. P, “Information Theory in Game Theory,” Entropy Basel Switz., vol. 20, no. 11, 
Oct. 2018, doi: 10.3390/e20110817.
[5] D. Lokshtanov and B. Subercaseaux, “Wordle is NP-hard,” ArXiv220316713 Cs, Mar. 2022, 
Accessed: Apr. 12, 2022. [Online]. Available: http://arxiv.org/abs/2203.16713
[6] B. J. Anderson and J. G. Meyer, “Finding the optimal human strategy for Wordle using 
maximum correct letter probabilities and reinforcement learning,” ArXiv220200557 Cs, Feb. 2022, 
Accessed: Apr. 12, 2022. [Online]. Available: http://arxiv.org/abs/2202.00557
[7] “Wordle - A daily word game.” https://www.nytimes.com/games/wordle (accessed Apr. 12, 
2022).
[8] A. Ratnaparkhi, “Maximum entropy models for natural language ambiguity resolution,” phd, 
University of Pennsylvania, USA, 1998.