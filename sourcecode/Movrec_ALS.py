
import sys
import itertools
from math import sqrt

import numpy as np
from numpy import matrix
from numpy.random import rand

from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext

import time

start = time.time()
# regularization constant
LAMBDA = 0.001   


def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]


def updatemovie(u,ratings, mvmat):
    """
    fixing movie features matrix and updating user features matrix
    """
    a = mvmat.T * mvmat
    YTr = mvmat.T * ratings[u, :].T
    #print (len(a))
    I = np.identity(len(a))
    b= LAMBDA * I
    #print (b.shape)
    a =  a+b
    
    return np.linalg.solve(a, YTr)


def RMSEcal(R, mvmat, umat):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    Rprime = np.dot(mvmat,umat.T)
    err = R - Rprime
    errsq = np.power(err, 2)
    mean = (np.sum(errsq))/(M * N)
    RMSE= np.sqrt(mean)
    
    return RMSE
	
	
def updateuser(m, usmat, ratings):
    """
    fixing user features matrix and updating movie features matrix
    """
    a = usmat.T * usmat
    Ytr = usmat.T * ratings[m, :].T
    
    I = np.identity(len(a))
    b= LAMBDA * I
    
    a =  a + b
    
    return np.linalg.solve(a, Ytr)

np.random.seed(0)	
	
	
if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print ("Usage: spark-submit Movierec_ALS.py path of ratings file</user/input_files/>")
        sys.exit(1)
    print("running Movie Recommendation system - USING ALS")

    conf = SparkConf().setAppName("Movie Rec ALS").set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)
    
    # load ratings and movie titles

    #movieLensHomeDir = "/user/input_files"
    movieLensHomeDir = sys.argv[1]

    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = sc.textFile(join(movieLensHomeDir, "ratings.txt")).map(parseRating)

    # movies is an RDD of (movieId, movieTitle)
    movies = dict(sc.textFile(join(movieLensHomeDir, "movies.txt")).map(parseMovie).collect())
    ratingscount = ratings.count()
    userscount = 6040
    moviescount = 3706
	
    print ("dataset contains ratings of %d users on %d movies." % (userscount, moviescount))
    
    Finratings = ratings.values().cache().collect()
    
    
    #converting ratings in RSS to matrix format : rdd->list -->array --> matrix
    r =Finratings #.collect()
    ra = np.array(r)
    rows, row_pos = np.unique(ra[:, 0], return_inverse=True)
    cols, col_pos = np.unique(ra[:, 1], return_inverse=True)
    rmat = np.zeros((len(rows), len(cols)), dtype=ra.dtype)
    rmat[row_pos, col_pos] = ra[:, 2]
    
    Ratmatrix = np.matrix(rmat)
    
	# number of users
    M =  userscount
	# number of movies
    N =  moviescount 
	#hidden latent features
    K =  15
    
    ITERATIONS = 20
    

    R = Ratmatrix # Rating matrix
    
    #  an initial matrix of dimension M x K 
    moviefmat = matrix(rand(M, K)) 
	#  an initial matrix of dimension N x K 
    userfmat = matrix(rand(N, K))
    
	# Broadcasting the Matrices
    Rb = sc.broadcast(R)
    mmat = sc.broadcast(moviefmat)
    umat = sc.broadcast(userfmat)
    
	#partitions variable
    p = 8
	
    for i in range(ITERATIONS):
        
		# funtion called to train one matrix , fixing the other
        moviefmat = sc.parallelize(range(M), p).map(lambda x: updatemovie(x,Rb.value, umat.value )).collect()
        
        # converting rdd data to matrix 
        moviefmat = matrix(np.array(moviefmat)[:, :, 0])
        # Broadcasting the new calculated matrix
        mmat = sc.broadcast(moviefmat)
        
		# function called to train one matrix , fixing the other
        userfmat = sc.parallelize(range(N), p).map(lambda x: updateuser(x, mmat.value, Rb.value.T)).collect()
        
		# converting rdd data to matrix 
        userfmat = matrix(np.array(userfmat)[:, :, 0])
        # Broadcasting the new calculated matrix
        umat = sc.broadcast(userfmat)
        
		# calculating RMSE value
        rmseval = RMSEcal(R, moviefmat, userfmat)
		
        print ("iteration no  = %d and RMSE in this iteration = %5.4f \n" % (i, rmseval) )
    
	#final rating matrix predicted by fixing the two matrix
    final_ratings=np.dot(moviefmat,userfmat.T)
	
    # Matrix to check 0 rating of user and do predictions
    Rwt = Ratmatrix>0.5 
    Rwt = Rwt.astype(np.float64)
	#output file
    outputfile=open("recomendations.txt",'w')
    flag=0
    
	#generating predictons to output file
    for i in range(M):
        flag=0
        recoutput="\n \t*********\tRecommended movies for user: "+"  "+str(i)+"\t ***********"+"\n"
        for j in range(N):
            if((Rwt[i,j]==0 and final_ratings[i,j]>3)):
		try:
                   
		   flag=1
                   recoutput=recoutput+"Movie title"+" :  "+movies[j]+","+"\t Movie ID"+" :  "+str(j)+","+"\t Predicted Rating: "+str(final_ratings[i,j])+"\n"
                except:
                   print("Exception")
	result=recoutput.encode('utf-8')
        if flag==1:
            outputfile.write(result)

    end = time.time()
    print("total running time = %5.4f" % ((end - start)/60))
    print("** Recomendations are generated. check recommendations.txt for recomendations of every user **")