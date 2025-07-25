import pandas as pd
import psycopg2
import pyodbc
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import text

def maximum(a, b):
     
    if a >= b:
        return a
    else:
        return b
     
# Driver code
a = 22
b = 4
print(maximum(a, b))