# FILE DESCRIPTION: This file is designed to plot the posterior probabilites saved to
# probabilities.txt by the last time the program ran. This is useful for "zooming in" on
# areas of interst after the initial calculations. We implemented this feature because
# we we're curious to compare the graph data directly with the multiple sequnce alignemnt. 

import numpy as np
import matplotlib.pyplot as plt

# Load probabilities.txt 
probabilities = np.loadtxt("probabilities.txt")
plt.plot(probabilities[0, :])
# Set axis range 
#plt.ylim(0.5, 1)
#plt.xlim(200000, 300000)
plt.title("Conservation Scores")
plt.xlabel("Base Position")
plt.ylabel("Posterior Probability")
plt.savefig("probabilitiesdetail.png")