import pandas as pd
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
