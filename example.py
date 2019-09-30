import numpy as np
import pandas

from sklearn.datasets import load_boston

import mysql.connector
import PCAscan


# example without SQL communication
boston=load_boston()


# we will do dimensionality reduction, not prediction
# add target vector to data matrix
boston.feature_names=list(boston.feature_names)+['MEDV']
boston.data=np.concatenate((boston.data,boston.target.reshape(-1,1)),axis=1)
PCAscan.dataLoss(boston.data)

# connection to SQL
database='Name of the database of interest'
table='Table name' # make sure table only contains numerical data 

server=PCAscan.Connection(host,username,password)
server.cursor.execute('use {}'.format(database))

matrix=server.SQLToMatrix(database,table)
PCAscan.dataLoss(matrix)


