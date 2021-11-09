from sqlalchemy import create_engine
import pandas as pd
import mysql.connector

engine = create_engine('mysql+mysqlconnector://root:Blokfluit23!@localhost/BDSE')


def put_in_df(table_name, dataframe):
        dataframe.to_sql(table_name, engine, if_exists='replace', index=False, chunksize=100)

def get_df():
        try:
                conn = mysql.connector.connect(
                        host="localhost",
                        user="root",
                        password="Blokfluit23!",
                        database='BDSE',
                )
                cur = conn.cursor()

                # execute stored procedure
                cur.callproc('get_data')

                # fetch all results or use fetchone()
                for res in cur.stored_results():
                        result = res.fetchall()
                return pd.DataFrame( result, columns=['Review', 'Label'])
        finally:
                cur.close()
                conn.close()