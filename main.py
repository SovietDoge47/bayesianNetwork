
def main():
    # Import required packages
    import math
    from pomegranate import *

    difficulty = DiscreteDistribution({'d0': 0.6, 'd1': 0.4})

    intelligence = DiscreteDistribution({'i0': 0.7, 'i1': 0.3})

    grade = ConditionalProbabilityTable(
        [['i0', 'd0', 'g1', 0.3],
         ['i0', 'd0', 'g2', 0.4],
         ['i0', 'd0', 'g3', 0.3],
         ['i0', 'd1', 'g1', 0.05],
         ['i0', 'd1', 'g2', 0.25],
         ['i0', 'd1', 'g3', 0.7],
         ['i1', 'd0', 'g1', 0.9],
         ['i1', 'd0', 'g2', 0.08],
         ['i1', 'd0', 'g3', 0.02],
         ['i1', 'd1', 'g1', 0.5],
         ['i1', 'd1', 'g2', 0.3],
         ['i1', 'd1', 'g3', 0.2]], [difficulty, intelligence])

    letter = ConditionalProbabilityTable(
        [['g1', 'l0', 0.1],
         ['g1', 'l1', 0.9],
         ['g2', 'l0', 0.4],
         ['g2', 'l1', 0.6],
         ['g3', 'l0', 0.99],
         ['g4', 'l1', 0.01]], [grade])
    )

    d1 = State(difficulty, name="difficulty")
    d2 = State(intelligence, name="intelligence")
    d3 = State(grade, name="grade")
    d4 = State(letter, name="letter")

    # Building the Bayesian Network
    network = BayesianNetwork("Solving the Monty Hall Problem With Bayesian Networks")
    network.add_states(d1, d2, d3, d4)
    network.add_edge(d1, d3)
    network.add_edge(d2, d3)
    network.add_edge(d3, d4)
    network.bake()

    beliefs = network.predict_proba({'letter': 'l1'})
    beliefs = map(str, beliefs)
    print("n".join("{}t{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
