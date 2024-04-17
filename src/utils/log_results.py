import pandas as pd

def read_log_results(file, remove_bg=True):
    """Reads the log.txt file and returns a pandas dataframe with the results for each category."""

    # Read txt file and extract the results
    with open(file, "r") as f:
        rows = []
        txt = f.readlines()
        for line in txt:
            sline = line.split("|")
            if len(sline) == 5:
                cat = sline[1].strip()
                try:
                    dice = float(sline[2].strip())
                    acc = float(sline[3].strip())
                    rows.append((cat, dice, acc))
                    
                except:
                    pass
        
    # Create pandas dataframe with the results
    df = pd.DataFrame(rows, columns=["cat", "dice", "acc"])
    
    # Remove background
    if remove_bg:
        return df[~(df['cat'] == "background")].reset_index(drop=True)
            
    return df
