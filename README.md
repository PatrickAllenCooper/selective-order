# Selective Order
Intended to establish a reasonable baseline for comparison via selective injection.

The idea is to present maximally difficult examples in rapid succession. In this way we hope to establish
 boundary conditions around the problem space.

Intuitively this will allow for superior performance in early performance and specualively superior 
performance overall across the problem space.

Independent experiments for this software apparatus include the distribution which 
determines model cycle and epochs per cycle, the attack type used, the epsilon used for the chosen atack,
 and the transfer method used.


### Paper and Justification
This consists of an intermediate paper discussing the merits etc. of    
Please reference the following link for further documentation.


### Running the examples
All examples perform the same basic operations:

1. Compute a baseline model.
2. Use that model to compute adversarial examples.
3. Selectively operate and train on those examples and note the effect on training time,
etc.
4. Log the results.

For each example there is a single GPU version and multi GPU version, these versions are discrete for 
stability purposes.


## Future Experiments


### Functional Adversarial Generation

#### Prompt
Complete experiments whereby new adversarial examples are generated each created either
epoch or each model cycle using the model trained to whatever extent it is at that point.
Compare and contrast their relative performance. 

#### Scheduled Date for Completion
August 8, 2021
