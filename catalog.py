#!python3
# coding: utf-8
"""Import data from catalogue into database"""


import pandas as pd
from sqlalchemy import create_engine


def chunky(data, table, n=100, k=1000):
    """Save data into database"""
    N = len(data)
    Nk = N % k
    for i in range(Nk):
        sub = data[i*Nk:(i+1)*Nk]
        sub.to_sql(table, eng,
                   if_exists="append",
                   chunksize=n)
        print(".", end="", flush=True)
    fin = data[Nk:]
    fin.to_sql(table, eng,
               if_exists="append",
               chunksize=n)
    print("!", end="\n", flush=True)
    return


eng = create_engine("sqlite:///lamost.db")

alls = pd.read_csv("catalog/dr5_v1.csv.gz",
                   sep="|", na_values=[-9999.00, "NULL"],
                   low_memory=False)

chunky(alls, "alls")

del alls

print("Done!")
