import datetime

def standardize_datetime(df_datetime):
    """
    Convert any datetime object into a standardized format : YYYY-MM-DD HH:MM
    - Handles cases where hours or minutes are single digit
    - Ensures that all formats are correctly aligned for comparison
    """
    
    if isinstance(df_datetime, str):
        try:
            df_datetime = datetime.datetime.strptime(df_datetime, "%Y-%m-%d %H:%M")
        except ValueError:
            try:
                df_datetime = datetime.datetime.strptime(df_datetime, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
    
    return df_datetime.strftime("%Y-%m-%d %H:%M")
