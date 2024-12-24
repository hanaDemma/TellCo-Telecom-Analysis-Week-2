import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os
# Add the correct path to the 'scripts' folder
sys.path.append(os.path.abspath("../scripts"))
# Function to get database connection
def get_engine():
    username = "postgres"
    password = "admin"
    hostname = "localhost"
    database = "XDR_Data"
    
    connection_string = f"postgresql+psycopg2://{username}:{password}@{hostname}:5432/{database}"
    return create_engine(connection_string)



# Query to load data
query = "SELECT * FROM xdr_data;"

# Load data into DataFrame
engine = get_engine()
xdr_data = pd.read_sql(query, engine)

# Sidebar navigation
# st.sidebar.title("Navigation")
st.sidebar.header('Telecommunication')
st.sidebar.write("Telecom dataset analyzed as follows:")
options = st.sidebar.selectbox('Select an Analysis', ['Exploratory Data Analysis', 'User Overview Analysis', 'User Engagement Analysis', 'User Experience Analysis', 'User Satisfaction Analysis'])

# Show initial telecom dataset
st.success("Initial Telecom Dataset")
st.write(xdr_data.head())
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



def aggregate_cluster_scores(data):
  """
  Aggregates the average satisfaction and experience scores per cluster.

  Args:
    data: The input DataFrame containing user data.

  Returns:
   A DataFrame with aggregated cluster scores.
  """

  cluster_stats = data.groupby('engagement_experience_segment')[['satisfaction_score', 'experience_score']].mean()
  return cluster_stats


# Display content based on the selected option
if options == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.success("Describe all telecom numerical values")
    st.write(xdr_data.describe())
    st.subheader("Check datatype of extracted dataset")
    dataType_of_Dataset = st.selectbox("Choose a column to view its datatype", xdr_data.columns)
    if st.button("View Datatype"):
       st.write(xdr_data[dataType_of_Dataset].dtypes)

elif options == "User Overview Analysis":
    # st.title("User Overview Analysis")
    # st.write("Here you can perform user overview analysis.")
    
    st.title("User Overview analysis")
    top_handsets = xdr_data['Handset Type'].value_counts().head(10)
    top_manufacturers = xdr_data['Handset Manufacturer'].value_counts().head(3)
    st.success('Top Ten handset types')
    st.write(top_handsets)
    st.bar_chart(top_handsets)
    st.success('Top three handset manufacturers')
    st.write(top_manufacturers)
    st.bar_chart(top_manufacturers)

    st.subheader('The top 5 handset types per top 3 handset manufacturer')
    filtered_data = xdr_data[xdr_data['Handset Manufacturer'].isin(top_manufacturers.index)]
    for h_manufacturer in top_manufacturers.index:
        fig, ax = plt.subplots(figsize=(18, 6))
        top_5_handsets_per_manufacturer = filtered_data[filtered_data['Handset Manufacturer'] == h_manufacturer]['Handset Type'].value_counts().head(5)
        st.success(h_manufacturer)
        st.write(top_5_handsets_per_manufacturer)
        st.write(sns.barplot(top_5_handsets_per_manufacturer))
        st.pyplot(fig)

    st.subheader('Aggregate Each Application per User')
    aggregated_xdr_data = aggregate_xdr_data(xdr_data)
    st.dataframe(aggregated_xdr_data)

    st.subheader('Exploratory Data Analysis (EDA) on Aggregated Application per User Data')
    st.dataframe(aggregated_xdr_data.describe())

    st.subheader('Variable transformations')
    st.success('Segment the users into the top five decile classes and Calculate Total Data per Decile Class')
    
    # from scripts.telecom_analysis import segment_users_and_calculate_total_data
    total_data_per_decile = segment_users_and_calculate_total_data(xdr_data)
    st.bar_chart(total_data_per_decile)

    st.subheader("Univariate Analysis: Dispersion Parameters")
    # from scripts.telecom_analysis import compute_dispersion_parameters
    dispersion_results = compute_dispersion_parameters(xdr_data)
    st.write(dispersion_results)

    st.subheader('Univariate Analysis: Graphical Dispersion Parameters Analysis')
    def graphicalDispersionParametersAnalysis(applications):
        for application in applications:
            st.success(application)
            st.bar_chart(dispersion_results[application])

    graphicalDispersionParametersAnalysis(['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 
                                           'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Youtube DL (Bytes)', 
                                           'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Google DL (Bytes)', 
                                           'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)', 'Gaming DL (Bytes)', 
                                           'Gaming UL (Bytes)', 'Other UL (Bytes)', 'Other DL (Bytes)'])

    st.subheader('Bivariate Analysis:')
    st.success('Using correlation between applications and Total DL and UL')
    
    def bivariateAnalysisbetweenApplicationAndTotalDL_UL():
        fig, ax = plt.subplots(figsize=(20, 8))
        correlation_matrix = xdr_data[['Total_DL_+_UL', 'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
                                      'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
                                      'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)', 
                                      'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other UL (Bytes)', 'Other DL (Bytes)']].corr()
        st.write(sns.heatmap(correlation_matrix, annot=True, cmap='viridis'))
        st.pyplot(fig)
    
    bivariateAnalysisbetweenApplicationAndTotalDL_UL()

    st.subheader('Correlation Analysis:')
    st.success('Computing a Correlation Matrix for Each Applications')
    # from scripts.telecom_analysis import correlationBetweenApplication
    applicationData = correlationBetweenApplication(xdr_data)

    def correlationBetweenEachApplications():
        fig, ax = plt.subplots()
        correlation_matrix_app = applicationData[['Social Media Data', 'Google Data', 'Email Data', 'YouTube Data', 
                                                  'Netflix Data', 'Gaming Data', 'Other Data']].corr()
        st.write(sns.heatmap(correlation_matrix_app, annot=True, cmap='viridis'))
        st.pyplot(fig)

    correlationBetweenEachApplications()

    st.subheader('Principal Component Analysis (PCA):')
    st.success('For Dimensionality Reduction')
    # Select only the desired columns
    data_selected = applicationData[['Social Media Data', 'Google Data', 'Email Data', 'YouTube Data', 
                                     'Netflix Data', 'Gaming Data', 'Other Data']]

    data_standardized = (data_selected - data_selected.mean()) / data_selected.std()

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_standardized)

    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    final_df = pd.concat([principal_df, xdr_data], axis=1)

    st.dataframe(final_df)


elif options == "User Engagement Analysis":
    st.title("User Engagement Analysis")
    aggregated_data_user_engagement, cluster_stats_user_engagement, top_10_most_engaged_users,normalized_data,engagement_clusters = analyze_user_engagement(xdr_data)

    st.subheader('Aggregate session metrics (session freq, session duration and total session) per customer')
    st.dataframe(aggregated_data_user_engagement.head())

    st.success('Top 10 most engaged users per application')
    st.dataframe(top_10_most_engaged_users.head())

    st.subheader('After Aggregated, Segment users into three clusters')
    cluster_one = aggregated_data_user_engagement[aggregated_data_user_engagement['clusters'] == 0]
    cluster_two = aggregated_data_user_engagement[aggregated_data_user_engagement['clusters'] == 1]
    cluster_three = aggregated_data_user_engagement[aggregated_data_user_engagement['clusters'] == 2]

    st.success('User Engagement First Cluster')
    st.write(cluster_one)

    st.success('User Engagement Second Cluster')
    st.write(cluster_two)

    st.success('User Engagement Third Cluster')
    st.write(cluster_three)

    st.subheader('Segmented user into clusters:')
    st.success('Cluster metrics')
    st.dataframe(cluster_stats_user_engagement.head())

    top_frequency = aggregated_data_user_engagement['Bearer Id'].nlargest(10)
    top_duration = aggregated_data_user_engagement['Dur. (ms)'].nlargest(10)
    top_traffic = aggregated_data_user_engagement['Total_DL_+_UL'].nlargest(10)

    st.success('Top Ten sessions frequencies')
    st.write(top_frequency)
    st.bar_chart(top_frequency)
    st.success('Top Ten sessions durations')
    st.write(top_duration)
    st.success('Top Ten sessions traffics')
    st.write(top_traffic)
    st.bar_chart(top_traffic)

    st.success('Top Three most used Applications')
    top_three_apps = xdr_data[['Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 
                               'Email UL (Bytes)', 'Youtube UL (Bytes)', 'Youtube DL (Bytes)', 
                               'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 
                               'Gaming UL (Bytes)', 'Other UL (Bytes)']].sum().nlargest(3)
    st.write(top_three_apps)
    st.bar_chart(top_three_apps)



# elif options == "User Experience Analysis":
#     st.title("User Experience Analysis")
#     # Add your analysis code here for User Experience
#     st.title('Experience Analytics')
#     st.success('Aggregate average TCP,RTT,TP and Handset type per user')
#     aggregated_average_experience_analysis,experience_clusters = aggregate_average_xdr_data(xdr_data)

#     st.dataframe(aggregated_average_experience_analysis)


#     # Handset Type Distribution
#     st.success("Handset Type Distribution")
#     handset_counts = aggregated_average_experience_analysis['Handset Type'].value_counts()
#     st.bar_chart(handset_counts)

#     st.success('Top, Bottom, and Most frequent values for TCP, RTT, and throughput')


#     # Find top, bottom, and most frequent values for TCP, RTT, and throughput
#     top_tcp_DL, bottom_tcp_DL, frequent_tcp_DL = find_top_bottom_frequent(aggregated_average_experience_analysis, 'TCP DL Retrans. Vol (Bytes)')
#     top_tcp_UL, bottom_tcp_UL, frequent_tcp_UL = find_top_bottom_frequent(aggregated_average_experience_analysis, 'TCP UL Retrans. Vol (Bytes)')
#     top_rtt_DL, bottom_rtt_DL, frequent_rtt_DL = find_top_bottom_frequent(aggregated_average_experience_analysis, 'Avg RTT DL (ms)')
#     top_rtt_UL, bottom_rtt_UL, frequent_rtt_UL = find_top_bottom_frequent(aggregated_average_experience_analysis, 'Avg RTT UL (ms)')
#     top_throughput_DL, bottom_throughput_DL, frequent_throughput_DL = find_top_bottom_frequent(aggregated_average_experience_analysis, 'Avg Bearer TP DL (kbps)')
#     top_throughput_UL, bottom_throughput_UL, frequent_throughput_UL = find_top_bottom_frequent(aggregated_average_experience_analysis, 'Avg Bearer TP UL (kbps)')
#     st.success('Top TCP')
#     st.write(top_tcp_DL,top_tcp_UL)
#     st.success('Bottom TCP')
#     st.write(bottom_tcp_DL,bottom_tcp_UL)
#     st.success('Frequent TCP')
#     st.write(frequent_tcp_DL,frequent_tcp_UL)
#     st.success('Top RTT')
#     st.write(top_rtt_DL,top_rtt_UL)
#     st.success('Bottom RTT')
#     st.write(bottom_rtt_DL,bottom_rtt_UL)
#     st.success('Frequent RTT')
#     st.write(frequent_rtt_DL,frequent_rtt_UL)
#     st.success('Top Throughput')
#     st.write(top_throughput_DL,top_throughput_UL)
#     st.success('Bottom Throughput')
#     st.write(bottom_throughput_DL,bottom_throughput_UL)
#     st.success('Frequent Throughput')
#     st.write(frequent_throughput_DL,frequent_throughput_UL)

#     st.subheader('The distribution of average Throughput per Handset type')
# def analyze_handset_throughput_metrics(data,avg_throughput):
#     avg_retransmission_by_handset = data.groupby('Handset Type')[avg_throughput].mean()
#     st.success("Average Throughput per Handset Type")
#     st.write(avg_retransmission_by_handset)
#     analyze_handset_throughput_metrics(xdr_data,'Avg Bearer TP DL (kbps)')
#     analyze_handset_throughput_metrics(xdr_data,'Avg Bearer TP UL (kbps)')

#     st.subheader('Average TCP retransmission view per handset type')
#     def analyze_handset_retrasmission_metrics(data,tcp_retrans):
#         avg_retransmission_by_handset = data.groupby('Handset Type')[tcp_retrans].mean()
#     st.success("Average TCP Retransmission per Handset Type")
#     st.write(avg_retransmission_by_handset)

#     analyze_handset_retrasmission_metrics(xdr_data,'TCP DL Retrans. Vol (Bytes)')
#     analyze_handset_retrasmission_metrics(xdr_data,'TCP UL Retrans. Vol (Bytes)')

#     st.subheader('Aggregated experience Cluster analysis')
#     cluster_1 = aggregated_average_experience_analysis[aggregated_average_experience_analysis['clusters'] == 0]
#     cluster_2 = aggregated_average_experience_analysis[aggregated_average_experience_analysis['clusters'] == 1]
#     cluster_3 = aggregated_average_experience_analysis[aggregated_average_experience_analysis['clusters'] == 2]
#     st.success("First cluster of Aggregated experience")
#     st.write(cluster_1)
#     st.success("Second cluster of Aggregated experience")
#     st.write(cluster_2)
#     st.success("Third cluster of Aggregated experience")
#     st.write(cluster_3)

#     st.title("Satisfaction Analysis")
#     st.success("Assign engagement and experience scores to users to calculate user satisfaction")
#     data_with_scores = assign_engagement_experience_scores(xdr_data, aggregated_data_user_engagement,aggregated_average_experience_analysis)

#     st.write(data_with_scores)

#     st.subheader('Calculated satisfaction score based on the average of engagement and experience scores')
#     st.write('Data with satisfaction score')
#     data_with_satisfaction = calculate_satisfaction_score(xdr_data)
#     st.write(data_with_satisfaction)
#     st.success('Top 10 satisfied customers')
#     top_satisfied_customers = find_top_satisfied_customers(data_with_satisfaction, 10)
#     st.write(top_satisfied_customers)
#     st.subheader('Build a regression model to predict customer satisfaction scores based on engagement and experience')
#     model, r2, mse = build_regression_model(xdr_data)
#     st.write("R-squarea and MSE of Regression Model")
#     st.write("R-squared:", r2)
#     st.write("Mean Squared Error:", mse)

#     st.subheader("Make predictions")
# def getEngagementAndExperienceScore(engagement_score,experience_score):
#     new_user_data = pd.DataFrame({'engagement_score': [engagement_score],
#                              'experience_score': [experience_score]})
#     return new_user_data
# engagement_score=st.number_input('engagement_score')
# experience_score=st.number_input('experience_score')
# # Make predictions using the trained model
# # st.write("The predicted satisfaction score of engagement_score = 0.8 and experience_score = 0.5")
# if st.button("Predict satisfaction score"):
#     new_user_data=getEngagementAndExperienceScore(engagement_score,experience_score)
#     predicted_satisfaction_score = model.predict(new_user_data)
#     st.success(f"The Predicted satisfaction score of {engagement_score} and {experience_score} is {predicted_satisfaction_score}")
#     # st.write("Predicted satisfaction score:", predicted_satisfaction_score)

#     st.subheader("Segment users into two clusters based on engagement and experience scores using k-means clustering")
#     segmented_data = segment_users_k_means(xdr_data)
#     cluster_segmented_1=segmented_data[segmented_data['engagement_experience_segment']==0]
#     cluster_segmented_2=segmented_data[segmented_data['engagement_experience_segment']==1]
#     st.success("First cluster based on engagement and experience scores")
#     st.write(cluster_segmented_1)
#     st.success("Second cluster based on engagement and experience scores")
#     st.write(cluster_segmented_2)

#     st.subheader("The average satisfaction and experience scores for each of the two clusters")
#     cluster_scores = aggregate_cluster_scores(segmented_data)
#     st.write(cluster_scores)





   
