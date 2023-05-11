import datetime

def datetime_now_str() :
    #datetime returns in the format: YYYY-MM-DD HH:MM:SS.millis but ':' is not supported for Windows' naming convention.
    dateTimeNowStr = str(datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S"))
    dateTimeNowStr = dateTimeNowStr.replace(':','.')
    dateTimeNowStr = dateTimeNowStr.replace(' ','.')
    return dateTimeNowStr