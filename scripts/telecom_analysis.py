import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def aggregate_xdr_data(data):
    """Aggregates xDR data per user and application.

    Args:
        The xDR data.

    Returns:
        The aggregated xDR data.

    """
    agg_xdr_data=pd.DataFrame(data)
    agg_xdr_data['Total_DL_and_UL_data'] = agg_xdr_data['Total DL (Bytes)'] + agg_xdr_data['Total UL (Bytes)']
    agg_xdr_data['Social Media Data'] = agg_xdr_data['Social Media DL (Bytes)']+agg_xdr_data['Social Media UL (Bytes)']
    agg_xdr_data['Google Data'] = agg_xdr_data['Google DL (Bytes)']+agg_xdr_data['Gaming UL (Bytes)']
    agg_xdr_data['Email Data']=agg_xdr_data['Email DL (Bytes)']+agg_xdr_data['Email UL (Bytes)']
    agg_xdr_data['YouTube Data']=agg_xdr_data['Email DL (Bytes)']+agg_xdr_data['Email UL (Bytes)']
    agg_xdr_data['Netflix Data']=agg_xdr_data['Netflix DL (Bytes)']+agg_xdr_data['Netflix UL (Bytes)']
    agg_xdr_data['Gaming Data']=agg_xdr_data['Gaming DL (Bytes)']+agg_xdr_data['Gaming UL (Bytes)']
    agg_xdr_data['Other Data'] = agg_xdr_data['Other DL (Bytes)']+agg_xdr_data['Other UL (Bytes)']

    columns = ['MSISDN/Number', 'Dur. (ms)','Bearer Id','Other Data','Gaming Data','Netflix Data','YouTube Data','Email Data', 'Google Data','Social Media Data', 'Total_DL_and_UL_data']

    df = agg_xdr_data[columns]

    # Aggregate data
    aggregated_df = df.groupby('MSISDN/Number').agg(
        Total_DL_And_UL=('Total_DL_and_UL_data', sum),# Total data (DL + UL)
        Total_Social_Media_Data=('Social Media Data',sum),
        Total_Google_Data=('Google Data', sum),
        Total_Email_Data=('Email Data',sum),
        Total_YouTube_Data=('YouTube Data', sum),
        Total_Netflix_Data=('Netflix Data',sum),
        Total_Gaming_Data=('Gaming Data', sum),
        Total_Other_Data=('Other Data',sum),
        Total_Session_Duration=('Dur. (ms)',sum),# Summing the session durations
        Total_xDR_Sessions=('Bearer Id',sum) ,# Counting the number of sessions
    )

    return aggregated_df


def segment_users_and_calculate_total_data(data):
  """
  Segments users into the top five decile classes based on total session duration and calculates the total data (DL+UL) per decile class.

  Args:
    data: The input DataFrame containing user information data.

  Returns:
    A DataFrame with decile class and total data per decile class.
  """

  # Calculate total DL and UL data per user
  data['Total_DL_+_UL'] = data['Total DL (Bytes)'] + data['Total UL (Bytes)']

  # Segment users into top five decile classes based on total session duration
  decile_labels = ['Decile 1', 'Decile 2', 'Decile 3', 'Decile 4', 'Decile 5']
  data['decile_class'] = pd.qcut(data['Dur. (ms)'], 5, labels=decile_labels)

  # Calculate total data per decile class
  total_data_per_decile = data.groupby('decile_class')['Total_DL_+_UL'].sum()

  return total_data_per_decile


def compute_dispersion_parameters(data):
  """
  Computes various dispersion parameters for a DataFrame.

  Args:
    data : The input DataFrame.

  Returns:
    A DataFrame containing dispersion parameters for each numeric column.
  """

  numeric_columns = data.select_dtypes(include='number').columns

  dispersion_params = pd.DataFrame(index=['Range', 'Variance', 'Std Dev', 'IQR', 'Coef Var'], columns=numeric_columns)

  for column in numeric_columns:
    dispersion_params.loc['Range', column] = data[column].max() - data[column].min()
    dispersion_params.loc['Variance', column] = data[column].var()
    dispersion_params.loc['Std Dev', column] = data[column].std()
    dispersion_params.loc['IQR', column] = data[column].quantile(0.75) - data[column].quantile(0.25)
    dispersion_params.loc['Coef Var', column] = data[column].std() / data[column].mean()

  return dispersion_params

def plot_dispersion_parameters(dispersion_results,applications):
    for application in applications:
      sns.barplot(data=dispersion_results[application])
      plt.title('Dispersion Parameters')
      plt.xlabel(application)
      plt.ylabel('Value')
      plt.xticks(rotation=45)
      plt.show()


def correlationBetweenApplication(data):
        agg_xdr_data = pd.DataFrame(data)
        agg_xdr_data['Social Media Data'] = agg_xdr_data['Social Media DL (Bytes)']+agg_xdr_data['Social Media UL (Bytes)']
        agg_xdr_data['Google Data'] = agg_xdr_data['Google DL (Bytes)']+agg_xdr_data['Gaming UL (Bytes)']
        agg_xdr_data['Email Data']=agg_xdr_data['Email DL (Bytes)']+agg_xdr_data['Email UL (Bytes)']
        agg_xdr_data['YouTube Data']=agg_xdr_data['Email DL (Bytes)']+agg_xdr_data['Email UL (Bytes)']
        agg_xdr_data['Netflix Data']=agg_xdr_data['Netflix DL (Bytes)']+agg_xdr_data['Netflix UL (Bytes)']
        agg_xdr_data['Gaming Data']=agg_xdr_data['Gaming DL (Bytes)']+agg_xdr_data['Gaming UL (Bytes)']
        agg_xdr_data['Other Data'] = agg_xdr_data['Other DL (Bytes)']+agg_xdr_data['Other UL (Bytes)']
        return agg_xdr_data;


def analyze_user_engagement(data):
    """
    Analyzes user engagement based on session metrics and segments users into clusters.

    Args:
      data: The input DataFrame containing user data.

    Returns:
      A DataFrame with segmented users and engagement metrics.
    """

    # Aggregate metrics per customer ID
    aggregated_data = data.groupby('MSISDN/Number').agg({'Bearer Id': 'sum',
                                                        'Dur. (ms)': 'sum',
                                                        'Total_DL_+_UL': 'sum'})

  
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(aggregated_data)

    normalized_data = pd.DataFrame(normalized_data)

    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(normalized_data)
    aggregated_data['clusters'] = clusters

    # Compute minimum, maximum, average, and total metrics per cluster
    cluster_stats = aggregated_data.groupby('clusters').agg(['min', 'max', 'mean', 'sum'])

    # Aggregate user total traffic per application
    traffic_per_app = data.groupby(['MSISDN/Number','Google DL (Bytes)','Google UL (Bytes)','Email DL (Bytes)','Email UL (Bytes)','Youtube UL (Bytes)','Youtube DL (Bytes)', 'Netflix DL (Bytes)','Netflix UL (Bytes)','Gaming DL (Bytes)','Gaming UL (Bytes)','Other UL (Bytes)'])['Total_DL_+_UL'].sum().reset_index()
    top_10_most_engaged_users = traffic_per_app.groupby(['MSISDN/Number','Google DL (Bytes)','Google UL (Bytes)','Email DL (Bytes)','Email UL (Bytes)','Youtube UL (Bytes)','Youtube DL (Bytes)', 'Netflix DL (Bytes)','Netflix UL (Bytes)','Gaming DL (Bytes)','Gaming UL (Bytes)','Other UL (Bytes)'])['Total_DL_+_UL'].sum().nlargest(10)

    return aggregated_data, cluster_stats, top_10_most_engaged_users,normalized_data,clusters


def aggregate_average_xdr_data(data):
    # Aggregate data
    aggregated_average_df = data.groupby('MSISDN/Number').agg({
                                                            'TCP DL Retrans. Vol (Bytes)':'mean',
                                                            'TCP UL Retrans. Vol (Bytes)':'mean',
                                                            'Avg RTT DL (ms)':'mean',
                                                            'Avg RTT UL (ms)':'mean',
                                                            'Avg Bearer TP DL (kbps)':'mean',
                                                            'Avg Bearer TP UL (kbps)':'mean',
                                                            'Handset Type':'first'
                                                        })
    select_columns=aggregated_average_df[ ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)','Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)','Avg Bearer TP UL (kbps)']]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(select_columns)

    normalized_data = pd.DataFrame(normalized_data)

    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(normalized_data)
    aggregated_average_df['clusters'] = clusters

    return aggregated_average_df,clusters


def find_top_bottom_frequent(data, column_name, n=10):
    """
    Finds the top n, bottom n, and most frequent values for a given column.

    Args:
        data: The input DataFrame.
        column_name: The name of the column to analyze.
        n: The number of values to find.

    Returns:
        A tuple containing the top n, bottom n, and most frequent values.
    """

    top_n = data[column_name].nlargest(n)
    bottom_n = data[column_name].nsmallest(n)
    most_frequent = data[column_name].value_counts().head(n)

    return top_n, bottom_n, most_frequent


def analyze_handset_throughput(data, throughput_column):
    """
    Analyzes the distribution of average throughput per handset type.

    Args:
        data (DataFrame): The input DataFrame containing handset data.
        throughput_column (str): The column name representing throughput.
    """
    avg_throughput_by_handset = data.groupby('Handset Type')[throughput_column].mean()

    print("\nAverage Throughput per Handset Type (in Mbps):\n")
    print(avg_throughput_by_handset.to_markdown())

def analyze_handset_retrasmission_metrics(data,tcp_retransmission):
  """
  Analyzes the distribution of average TCP retransmission view per handset type.

  Args:
    data The input DataFrame containing handset data.
  """

  avg_retransmission_by_handset = data.groupby('Handset Type')[tcp_retransmission].mean()


  print("\nAverage TCP Retransmission Count per Handset Type:\n")
  print(avg_retransmission_by_handset.to_markdown())

  
def assign_engagement_experience_scores(data, engagement_clusters,experience_clusters):
  """
  Assigns engagement and experience scores to users based on Euclidean distance.

  Args:
    data: The input DataFrame containing user data.
    engagement_clusters: The DataFrame with engagement clusters.
    experience_clusters: The DataFrame with experience clusters.

  Returns:
    A DataFrame with assigned engagement and experience scores.
  """
  engagement_clusters = engagement_clusters.drop('clusters', axis=1)
  experience_clusters = experience_clusters.drop(['clusters','Handset Type'], axis=1)

  engagement_distances = euclidean_distances(data[['Bearer Id','Dur. (ms)','Total_DL_+_UL']], engagement_clusters.iloc[0].values.reshape(1, -1))
  experience_distances = euclidean_distances(data[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)','Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)','Avg Bearer TP UL (kbps)']], experience_clusters.iloc[2].values.reshape(1, -1))

  data['engagement_score'] = engagement_distances.min(axis=1)
  data['experience_score'] = experience_distances.min(axis=1)

  return data

def calculate_satisfaction_score(data):
    """
    Calculates a satisfaction score based on engagement and experience scores.

    Args:
        data: The input DataFrame containing user data.

    Returns:
        A Series with satisfaction scores for each user.
    """

    data['satisfaction_score'] = (data['engagement_score'] + data['experience_score']) / 2
    return data

def find_top_satisfied_customers(data, n=10):
    """
    Finds the top n satisfied customers based on their satisfaction score.

    Args:
        data : The input DataFrame containing user data.
        n: The number of top customers to find.

    Returns:
         A Series with the top n customer IDs.
    """

    top_satisfied = data.nlargest(n, 'satisfaction_score')['MSISDN/Number']
    return top_satisfied



def build_regression_model(data):
  """
  Builds a regression model to predict satisfaction score.

  Args:
    data : The input DataFrame containing user data.

  Returns:
  A tuple containing the trained model, R-squared score, and mean squared error.
  """

  # Split data into features and target variable
  X = data[['engagement_score', 'experience_score']]
  y = data['satisfaction_score']

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create and train a linear regression model
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = model.predict(X_test)

  # Evaluate the model
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  return model, r2, mse


def segment_users_k_means(data):
  """
  Segments users into two clusters based on engagement and experience scores.

  Args:
    data: The input DataFrame containing user data.

  Returns:
    A DataFrame with segmented users.
  """

  # Select relevant columns
  engagement_experience_metrics = data[['engagement_score', 'experience_score']]

  # Standardize the data
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(engagement_experience_metrics)

  # Perform k-means clustering
  kmeans = KMeans(n_clusters=2, random_state=42)
  clusters = kmeans.fit_predict(scaled_data)

  # Add cluster labels to the original DataFrame
  data['engagement_experience_segment'] = clusters

  return data