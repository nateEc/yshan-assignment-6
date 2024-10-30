from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):

    # STEP 1
    # TODO 1: Generate a random dataset X of size N with values between 0 and 1
    # and a random dataset Y with normal additive error (mean mu, variance sigma^2).
    # Hint: Use numpy's random's functions to generate values for X and Y
    X = np.random.rand(N)  # Generate random values for X
    Y = X * 2 + np.random.normal(mu, np.sqrt(sigma2), N)  # Linear relationship with noise

    # TODO 2: Fit a linear regression model to X and Y
    # Hint: Use Scikit Learn
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)  # Fit the model
    slope = model.coef_[0]  # Extract slope from the fitted model
    intercept = model.intercept_  # Extract intercept from the fitted model

    # TODO 3: Generate a scatter plot of (X, Y) with the fitted regression line
    # Hint: Use Matplotlib
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, label="Data points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label=f"y = {slope:.2f}x + {intercept:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Scatter Plot of (X, Y) with Regression Line\ny = {slope:.2f}x + {intercept:.2f}")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Step 2: Run S simulations and create histograms of slopes and intercepts

    # TODO 1: Initialize empty lists for slopes and intercepts
    slopes = []
    intercepts = []

    # TODO 2: Run a loop S times to generate datasets and calculate slopes and intercepts
    for _ in range(S):
        X_sim = np.random.rand(N)  # Generate random X values with size N between 0 and 1
        Y_sim = X_sim * 2 + np.random.normal(mu, np.sqrt(sigma2), N)  # Generate Y values with normal additive error

        sim_model = LinearRegression()  # Initialize model
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)  # Fit model

        slopes.append(sim_model.coef_[0])  # Append slope
        intercepts.append(sim_model.intercept_)  # Append intercept

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of more extreme slopes and intercepts
    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
