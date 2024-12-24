### Task 1- User Overview Analysis 

This repository contains the code and analysis for telecommunications datasets. The project focuses on conducting a User Overview Analysis, performing Exploratory Data Analysis (EDA), and communicating useful insights to guide business decisions.
## Project Overview
# 1. Repository Setup
- Create a GitHub repository with a dedicated branch: task-1.
- Committ progress at least three times a day with descriptive messages.
# 2. User Overview analysis
The objective of this phase is to provide a comprehensive analysis of user behavior. Key tasks include:
  - Identifying top 10 handsets used by customers.
   - Identifying top 3 handset manufacturers
   - Identifying top 5 handsets per top 3 handset manufacturer
    # User behavior analysis 
    - Number of xDR sessions
    - Session duration
    - Total download (DL) and upload (UL) data
    - Total data volume (in Bytes)

# 3. Exploratory Data Analysis (EDA)
- Describe Variables: Outline relevant variables and their associated data types.
- Variable Transformations: Apply transformations if needed.
- Basic Metrics: Calculate basic statistics such as mean, median, standard deviation, etc.
- Non-Graphical Univariate Analysis: Explore individual variables numerically.
- Graphical Univariate Analysis: Visualize the distribution of individual variables.
- Bivariate Analysis: Explore relationships between two variables.
- Correlation Analysis: Identify correlations between different features.
- Dimensionality Reduction: Apply techniques like PCA if needed for analysis.

# How to Use
- Clone the Repository:
git clone https://github.com/hanaDemma/TellCo-Telecom-Analysis-Week-2.

- Switch to Task-1 Branch:

git checkout task-1

- Run the Notebook:

    - Install dependencies:

        pip install -r requirements.txt

    - Open and execute the Jupyter notebook.

# Key Files
- telecom_data_analysis.ipynb: Contains the analysis and visualizations
- requirements.txt: List of required Python libraries.
- README.md: Project documentation.

# Technologies Used
- Libraries:
    - pandas, matplotlib, seaborn: For data manipulation and visualization.
    - scikit-learn: For clustering (e.g., k-means).
    - sqlalchemy: For database interaction
# Task-2 User Engagement Analysis

This task focuses on analyzing user engagement metrics, including session frequency, session duration, and total traffic (download and upload).

# Objective
- Aggregate User Engagement:

    - Analyze each application's usage per user.
    - Normalize engagement metrics for consistency.
- Key Metrics:

    - Session Frequency: How often users engage with the network.
    - Session Duration: The average length of each session.
    - Session Traffic: Total traffic (download and upload) in bytes.
- EDA for Engagement Metrics:

    - Conduct an exploratory analysis of user engagement metrics.
    - Visualize the top 3 most-used applications using appropriate plots.
- Clustering:

    - Group users into k engagement clusters based on the engagement metrics (e.g., frequent users vs. occasional users).
# Development Instructions
- Create a task-2 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).
# Key Features
- Session Frequency: Measure how frequently users engage with the service.
- Session Duration: Analyze the duration of each user’s session.
- Traffic Metrics: Total download and upload data used by each user.
- K-Means Clustering: Use k-means clustering to segment users based on engagement patterns.

## Task-3 Experience Analytics
The objective of this phase is to conduct deep user exprience anlysis by focusing network parameters (TCP retransmission, Round Trip Time (RTT), Throughpu)

- Aggregate, per customer:
   - Average TCP retransmission
   - Average RTT
   - Handset type
   - Average throughput
- Compute & list 10 of the top, bottom, and most frequent:
    - TCP values in the dataset. 
    - RTT values in the dataset.
    - Throughput values in the dataset.

# Development Instructions
- Create a task-3 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).

## Task-4 Satisfaction Analysis
 
The objective of this phase to analyze user satifaction in depth Based on the engagement analysis + the experience analysis

# Development Instructions
- Create a task-4 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).

## 5. Dashboard Development 
The objective of this phase to develop and visualize dashboard using streamlit

# Development Instructions
- Create a task-5 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).