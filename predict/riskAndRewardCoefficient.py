import tensorflow as tf
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class RiskAndRewardCoefficient:
    def __init__(self, ticker, range, train_sample_size, stock_ranking, financialfactor_data):
        print('')
        self.ticker = ticker
        self.numPrediction = range
        self.trainSampleSize = train_sample_size
        self.stockRanking = stock_ranking
        self.financialfactorData = financialfactor_data

    def calculateCoefficient(self):
        stock_rank = self.stockRanking
        stock_peratio = self.financialfactorData

        print(stock_rank)
        print(stock_peratio)

        num_records = len(stock_peratio)

        # Plot generated house and size
        plt.plot(stock_rank, stock_peratio, "bx")  # bx = blue x
        plt.ylabel("Rank")
        plt.xlabel("PeRatio")
        #plt.show()
        print('test')

        # you need to normalize values to prevent under/overflows.
        def normalize(array):
            if (array.std() == 0):
                return array
            return (array - array.mean()) / array.std()


        num_train_samples = math.floor(num_records * self.trainSampleSize)

        train_rank = np.asarray(stock_rank[:num_train_samples])
        train_peratio = np.asanyarray(stock_peratio[:num_train_samples:])

        train_rank_norm = normalize(train_rank)
        train_peratio_norm = normalize(train_peratio)

        # define test data
        test_rank = np.array(stock_rank[num_train_samples:])
        test_peratio = np.array(stock_peratio[num_train_samples:])

        test_rank_norm = normalize(test_rank)
        tf_factor_norm = normalize(test_peratio)

        tf_rank = tf.placeholder("float", name="stock_rank_var")
        tf_factor = tf.placeholder("float", name="stock_coefficient_var")

        tf_pe_coff = tf.Variable([.3], name="peratio_coff")
        tf_rank_pred = tf.multiply(tf_pe_coff, tf_factor)
        sess = tf.Session()
        tf_cost = tf.reduce_sum(tf.pow(tf_rank_pred - tf_rank, 2)) / (2 * num_train_samples)
        learning_rate = 0.1
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)


        init = tf.global_variables_initializer()



        sess.run(init)
        for i in range(100):
            for (x, y) in zip(train_rank_norm, tf_factor_norm):
                sess.run(optimizer, feed_dict={tf_rank: x, tf_factor: y})

        print("Optimization Finished!")
        training_cost = sess.run(tf_cost, feed_dict={tf_rank: train_rank_norm, tf_factor: tf_factor_norm})
        print("Trained cost=", training_cost, "size_factor=", sess.run(tf_pe_coff), '\n')

