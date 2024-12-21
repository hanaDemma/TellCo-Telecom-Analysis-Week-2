import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

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