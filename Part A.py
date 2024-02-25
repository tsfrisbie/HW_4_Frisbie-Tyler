#Chat GPT helped with this code
#Started hw to late and was unable to find a partner

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_distributions():
    """
    Plot Probability Density Function (PDF) and Cumulative Distribution Function (CDF)
    for normal distributions N(0,1) and N(175,3).
    """
    # Define the parameters
    mu = 0
    sigma = 1
    mu2 = 175
    sigma2 = 3
    # Generate x values starting point, stopping point, and number of samples.
    x_values = np.linspace(-10, 10, 100)
    # Calculate the probability density function for the normal distributions
    pdf1 = stats.norm.pdf(x_values, mu, sigma)
    pdf2 = stats.norm.pdf(x_values, mu2, sigma2)
    # Calculate the cumulative distribution function for the normal distributions
    cdf1 = stats.norm.cdf(x_values, mu, sigma)
    cdf2 = stats.norm.cdf(x_values, mu2, sigma2)
    # Plot the results and define graph titles, x and y labels.
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(x_values, pdf1, label='N(0,1)')
    plt.title('PDF for N(0,1)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(x_values, pdf2, label='N(175, 3)')
    plt.title('PDF for N(175, 3)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(x_values, cdf1, label='N(0,1)')
    plt.title('CDF for N(0,1)')
    plt.xlabel('x')
    plt.ylabel('F(x) = integrate(-inf,x) f(x)dx')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(x_values, cdf2, label='N(175, 3)')
    plt.title('CDF for N(175, 3)')
    plt.xlabel('x')
    plt.ylabel('F(x) = integrate(-inf,x) f(x)dx')
    plt.legend()

    plt.tight_layout()
    plt.show()

def calculate_probabilities():
    """
    Calculate probabilities for the normal distributions.
    Returns:
        float: Probability P(x<1|N(0,1)).
        float: Probability P(x>μ+2σ|N(175, 3)).
    """
    # Define the parameters
    mu = 0
    sigma = 1
    mu2 = 175
    sigma2 = 3
    # Calculate the probability P(x<1|N(0,1))
    prob_x_lt_1 = stats.norm.cdf(1, mu, sigma)
    # Calculate the probability P(x>μ+2σ|N(175, 3))
    prob_x_gt_mu_plus_2sigma = 1 - stats.norm.cdf(mu + 2*sigma, mu2, sigma2)
    return prob_x_lt_1, prob_x_gt_mu_plus_2sigma

# Main program
if __name__ == "__main__":
    # Call the function to plot distributions
    plot_distributions()
    # Call the function to calculate probabilities
    prob_x_lt_1, prob_x_gt_mu_plus_2sigma = calculate_probabilities()
    # Print the probabilities
    print("P(x<1|N(0,1)): {:.5f}".format(prob_x_lt_1))
    print("P(x>μ+2σ|N(175, 3)): {:.5f}".format(prob_x_gt_mu_plus_2sigma))