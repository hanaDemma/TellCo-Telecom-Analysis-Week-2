def convertByteIntoMegaByte(data):
    # We Have to convert some the data into MB or TB or GB
    megabyte=1*10e+5
    data['Bearer Id']=data['Bearer Id']/megabyte
    data['IMSI']=data['IMSI']/megabyte
    data['MSISDN/Number']=data['MSISDN/Number']/megabyte
    data['IMEI']=data['IMEI']/megabyte
    for column in data.columns:
        if 'Bytes' in column:
            data[column]=data[column]/megabyte
    return data