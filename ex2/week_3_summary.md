* Logistic regression lets us use some of the techniques we already learned to partition inputs between a set of *discrete* possible outputs
* The sigmoid function expresses this idea by representing extreme tail-end probabilities with a smooth "transition" zone in the middle.
* Similarly the cost function now uses logarithms to capture the intuition of moving from a 0-cost point at one end (when we guess the correct classification with total confidence) to an infinite cost limit at the other end (when we guess the wrong classification with total confidence).
* Basic gradient descent turns out to be the same since in the end it's just minimizing a generic cost function -- our cost function has changed but the process of minimizing it hasn't
* Multi-class logistic regressions can be handled by converting them into a collection of binary regressions -- each class taking a turn to evaluate against the combination of *all* other classes -- and then solving them one at a time
