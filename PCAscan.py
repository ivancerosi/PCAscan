import numpy as np
import pandas

from sklearn.datasets import load_boston

import mysql.connector

def GaussElim(matrix,result):
    matrix=np.array(matrix).copy()
    result=np.array(result).copy()

    floatFunc=np.vectorize(lambda x:float(x))
    matrix=floatFunc(matrix)
    result=floatFunc(result)
    
    dim1=matrix.shape[0]

    def removeZero(matrix, result, exception=-1): # remove 0s on diagonals
        for x in range(dim1):
            if matrix[x,x]==0:
                for y in range(dim1): # find a row which can fill the 0 element
                    if matrix[y,x]!=0 and y!=exception:
                        matrix[x,:]+=matrix[y,:]
                        result[x,0]+=result[y,0]
                        break
        return matrix,result


    def eliminate(matrix,result): # create matrix where non-diagonal elements are 0
        dim1=matrix.shape[0] 

        for x in range(dim1):
            pivot=matrix[x,x]
            if pivot==0:
                matrix,result=removeZero(matrix,result,exception=x-1)
                pivot=matrix[x,x]

            for y in range(dim1): # start eliminating column values except for pivot
                if x==y or matrix[y,x]==0:
                    continue
                else:
                    multiplier=float(matrix[y,x]/pivot)
                    matrix[y,:]=matrix[y,:]*1.0-matrix[x,:]*multiplier
                    result[y,0]=result[y,0]-(result[x,0]*multiplier)
    
        return matrix, result

    solution=[]
    matrix,result=eliminate(matrix,result)
    for x in range(dim1):
        solution.append(result[x,0]/matrix[x,x])

    return np.array(solution).reshape(-1,1)



def VarCov(matrix):
    N=matrix.shape[0]
    f1=np.dot(np.transpose(matrix),matrix)

    meansVector=np.zeros([matrix.shape[1],1])
    for x in range(meansVector.shape[0]):
        meansVector[x]=np.sum(matrix[:,x])

    meansMatrix=np.dot(meansVector,np.transpose(meansVector))
    return (f1/N)-meansMatrix/(N**2)


def GramSchmidt(matrix):

    matrix=matrix.copy().astype('float64')
    Q=matrix.copy()
    for x in range(1,matrix.shape[0]):
        orig=matrix[:,x].copy()
        for y in range(1,x+1):
            Q[:,x]=Q[:,x]-np.dot(matrix[:,x],Q[:,x-y])/np.dot(Q[:,x-y],Q[:,x-y])*Q[:,x-y]

    for x in range(matrix.shape[0]):
        Q[:,x]=Q[:,x]/np.linalg.norm(Q[:,x])

    R=np.dot(np.transpose(Q),matrix)
    return Q,R

def eigenvalues(matrix,decimals=10):
    matrix=matrix.copy().astype('float64')
    Continue=True
    while (Continue==True):
        Q,R=GramSchmidt(matrix)
        matrix=np.dot(R,Q)
        Continue=False
        for i in range(matrix.shape[0]):
            for j in range(i+1,matrix.shape[0]):
                if np.round(matrix[j,i],decimals)!=0:
                    Continue=True
                    break
            if Continue==True:
                break
    eigenvalues=[]
    for x in range(matrix.shape[0]):
        eigenvalues.append(matrix[x,x])

    return np.array(eigenvalues)

def push(e,x):
    x.append(x[-1])
    for i in range(1,len(x)):
        x[-i]=x[-i-1]
    x[0]=e
    return x

def eigenvectors(matrix,decimals=10):
    eigenv=eigenvalues(matrix,decimals)

    matrix=matrix.copy()
    eigv=[]
    results=[]
    for eigen in eigenv:
        mat=matrix.copy()

        for x in range(mat.shape[0]):
            mat[x,x]-=eigen
        solution=np.zeros([matrix.shape[0],1])-mat[:,0].reshape(-1,1)

        results=list(GaussElim(mat[1:,1:],solution[1:,:]))

        results=np.array(push(np.array(1),results))
        results=results/np.linalg.norm(results)
        eigv.append(np.array(results))
    return np.transpose(np.array(eigv))






def name(obj,namespace):
    return [name for name in namespace if namespace[name] is obj][0]

class Connection:
    def __init__(self,host,user,pwd):
        self.db = mysql.connector.connect(host=host,user=user,password=pwd)
        self.cursor  = self.db.cursor()

    def createDatabase(self, name):
        self.cursor.execute('CREATE DATABASE {}'.format(name))

    def showDatabases(self):
        names=[]
        self.cursor.execute('SHOW DATABASES')
        for db in self.cursor:
            names.append(db[0])
        return names

    def createTableID(self, name, id_table):
        execute="CREATE TABLE {} (ID SMALLINT(5) UNSIGNED);".format(name)
        self.cursor.execute(execute)

        for row_id in id_table:
            self.cursor.execute("INSERT INTO {}(ID) VALUES ({})".format(name,row_id))

    def createTable(self, name, columns):
        columns=' double, '.join(columns)+' double'
        execute="CREATE TABLE {} ({})".format(name,columns)
        self.cursor.execute(execute)

    def MatrixToSQL(self, name, data):
        for x in data.data:
            x=[str(x) for x in x]
            x=','.join(x)
            execute='INSERT INTO {} VALUES ({})'.format(name,x)
            server.cursor.execute(execute)
            server.db.commit()

    def SQLToMatrix(self, db, table):
        self.cursor.execute('use {}'.format(db))
        self.cursor.execute('select * from {}'.format(table))
        query=self.cursor.fetchall()

        matrix = np.array(query)

        return matrix

        
    
    def showTables(self, *argv):
        self.cursor.execute('SHOW TABLES')
        for x in self.cursor:
            print(x)

    def addData(self, table, **kwargs):
        def SQLtype(array):
            if type(array.reshape(-1,1)[0,0])==type(int):
                return "INT"
            elif type(array.reshape(-1,1)[0,0])==type(np.int32):
                return "INT"
            elif type(array.reshape(-1,1)[0,0])==type(np.int64):
                 return "BIGINT"
            elif type(array.reshape(-1,1)[0,0])==float:
                return "FLOAT"
            elif type(array.reshape(-1,1)[0,0])==np.float32:
                return "FLOAT"
            elif type(array.reshape(-1,1)[0,0])==np.float64:
                return "DOUBLE"
            else:
                raise Exception("Attempting to add data type that is not int or float")
        for key,value in kwargs.items():
            value=np.array(value)
            print('ALTER TABLE `{}` ADD COLUMN (`{} {});'.format(table,key,SQLtype(value)))
            self.cursor.execute('ALTER TABLE `{}` ADD COLUMN ({} {});'.format(table,key,SQLtype(value)))
            for i,element in enumerate(value):
                self.cursor.execute('UPDATE `{}` SET {}={} WHERE ID={}'.format(table,key,element,i))
                #print('UPDATE `{}` SET {}={} WHERE ID={}'.format(table,key,element,i))
        self.cursor.execute('COMMIT')

    def loadData(self, table, cols=[]):
        if cols==[]:
            cols='*'
        else:
            cols=','.join(cols)
        
        self.cursor.execute('SELECT {} FROM {}'.format(cols,table,))

        return self.cursor.fetchall()

    def currentDB(self):
        self.cursor.execute('SELECT DATABASE()')
        x=self.cursor.fetchone()[0]
        return x
    
    def ColumnNames(self, table):
        db=self.currentDB()
        self.cursor.execute('SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME=\'{}\''.format(table))
        lista=[]
        for x in self.cursor:
            lista.append(x[3])
        self.cursor.execute('USE {}'.format(db))
        return lista

    def NumRows(self,table):
        self.cursor.execute('SELECT MAX(ID) FROM {}'.format(table))
        max_row=int(self.cursor.fetchone()[0])
        return max_row

    def close(self):
        self.cursor.close()
        self.db.close()






def SQLtype(array):
    if type(array.reshape(-1,1)[0,0])==type(int):
        return "INTEGER(MAX)"
    elif type(array.reshape(-1,1)[0,0])==type(np.int32):
        return "INTEGER(MAX)"
    elif type(array.reshape(-1,1)[0,0])==type(np.int64):
         return "BIGINT(MAX)"
    elif type(array.reshape(-1,1)[0,0])==float:
        return "FLOAT(MAX)"
    elif type(array.reshape(-1,1)[0,0])==np.float32:
        return "FLOAT(MAX)"
    elif type(array.reshape(-1,1)[0,0])==np.float64:
        return "DOUBLE(MAX)"
    else:
        raise Exception("Attempting to add data type that is not int or float")


def dataLoss(matrix):
    assert isinstance(matrix,np.ndarray),"Please make sure matrix is of type NumPy array"
    covMatrix = VarCov(matrix)
    eigenv = eigenvalues(covMatrix)
    print('DATA DIMENSIONALITY COMPRESSION POTENTIAL [PCA]\n******************************************')
    print('COMPONENTS RETAINED \t INFORMATION RETAINED')
    for x in range(0,matrix.shape[1]):
        if x==0:
            info_retained=1
        else:
            info_retained=1-np.sum(eigenv[-x:])/np.sum(eigenv)
        print('{}/{}:\t\t\t{}%'.format(matrix.shape[1]-x,matrix.shape[1],info_retained*100))



