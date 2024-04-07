import sys
import random
import numpy as np
import pandas as pd

def load_data(filename):
    # Load data from CSV file
    df = pd.read_csv(filename)
    return df

def epsilon_greedy(epsilon, data, threshold, train_percentage):
    num_arms = data.shape[1]
    num_rows = data.shape[0]
    train_rows = int(num_rows * (train_percentage / 100))
    
    # Initialize counts and sums for each arm
    counts = np.zeros(num_arms)
    successes = np.zeros(num_arms)

    for i in range(train_rows):
        if random.random() > epsilon:
            # Exploit: choose the best arm so far
            best_arm = np.argmax(successes / (counts + 1e-10))  # Add a small value to avoid division by zero
        else:
            # Explore: choose a random arm
            best_arm = random.randrange(num_arms)
        
        counts[best_arm] += 1
        if data.iloc[i, best_arm] < threshold:
            successes[best_arm] += 1
    
    success_probabilities = successes / counts
    
    # Choose the best arm for the rest of the data
    final_best_arm = np.argmax(success_probabilities)

    test_successes = 0
    for i in range(train_rows, num_rows):
        if data.iloc[i, final_best_arm] < threshold:
            test_successes += 1
    
    test_success_rate = test_successes / (num_rows - train_rows)
    
    return success_probabilities, final_best_arm, test_success_rate

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) != 5:
        print("ERROR: Not enough or too many input arguments.")
        sys.exit(1)
    
    filename = sys.argv[1]
    epsilon = float(sys.argv[2])
    train_percentage = int(sys.argv[3])
    threshold = float(sys.argv[4])
    data = load_data(filename)
    
    # Run the epsilon-greedy algorithm
    success_probabilities, final_best_arm, test_success_rate = epsilon_greedy(epsilon, data, threshold, train_percentage)
    
    # Report results
    print(f"Nanda Utkarsh A20544986 solution:")
    print(f"epsilon: {epsilon}")
    print(f"Training data percentage: {train_percentage} %")
    print(f"Success threshold: {threshold}\n")
    print("Success probabilities:")
    for i, label in enumerate(data.columns):
        print(f"P({label}) = {success_probabilities[i]:.4f}")
    print(f"\nBandit [{data.columns[final_best_arm]}] was chosen to be played for the rest of the data set.")
    print(f"{data.columns[final_best_arm]} Success percentage: {test_success_rate * 100:.2f}%")
