"""
Example Python script demonstrating code quality standards.

This script shows proper formatting, imports, and documentation
that will pass all CI checks.
"""

import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame containing the loaded data
    """
    return pd.read_csv(filepath)


def analyze_data(df: pd.DataFrame) -> Dict[str, float]:
    """
    Perform basic statistical analysis on the dataframe.

    Args:
        df: Input dataframe to analyze

    Returns:
        Dictionary containing statistical measures
    """
    stats = {
        "mean": df.select_dtypes(include=[np.number]).mean().mean(),
        "std": df.select_dtypes(include=[np.number]).std().mean(),
    }
    return stats


def plot_distribution(data: List[float], title: str = "Distribution") -> None:
    """
    Create a histogram plot of the data distribution.

    Args:
        data: List of numerical values to plot
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()


def main() -> None:
    """Main execution function."""
    print("Data Science Capstone - Example Script")
    print("This script demonstrates proper code quality standards.")

    # Example usage
    sample_data = np.random.normal(100, 15, 1000)
    plot_distribution(sample_data, "Normal Distribution Example")

    # Create a sample dataframe
    df = pd.DataFrame({"values": sample_data, "category": np.random.choice(["A", "B", "C"], 1000)})

    stats = analyze_data(df)
    print(f"Statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
