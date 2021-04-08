from pomegranate import *

# Define starting probabilities
start = DiscreteDistribution({
    "sun": 0.5,
    "rain": 0.5
})

# Define transition model
transitions = ConditionalProbabilityTable([
    ["sun", "sun", 0.8],
    ["sun", "rain", 0.2],
    ["rain", "sun", 0.3],
    ["rain", "rain", 0.7]
], [start])

# Create Markov chain
model = MarkovChain([start, transitions])

# Sample 50 states from chain
# Eg: ['sun', 'sun', 'rain', 'rain', 'sun', 'sun', 'sun', 'sun', 'sun', 'sun', 'sun', 'sun', 
# 'sun', 'sun', 'sun', 'sun', 'rain', 'sun', 'sun', 'rain', 'sun', 'sun', 'sun', 'sun', 'sun', 
# 'sun', 'sun', 'rain', 'rain', 'rain', 'rain', 'rain', 'rain', 'rain', 'rain', 'rain', 'sun', 
# 'sun', 'sun', 'rain', 'sun', 'sun', 'rain', 'rain', 'rain', 'rain', 'rain', 'rain', 'rain', 'rain']
print(model.sample(50))
