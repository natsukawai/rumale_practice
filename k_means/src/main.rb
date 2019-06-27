# ruby (rumale) implementation of the following page:
# https://pythondatascience.plavox.info/scikit-learn/%E3%82%AF%E3%83%A9%E3%82%B9%E3%82%BF%E5%88%86%E6%9E%90-k-means

require 'daru'
require 'rumale'

df = Daru::DataFrame.from_csv("../input/Wholesale_customers_data.csv")
pp df.head(5)

df.delete_vectors('Channel', 'Region')
pp df.head(5)

analyzer = Rumale::Clustering::KMeans.new(n_clusters: 4, max_iter: 50, random_seed: 1)
samples = Numo::DFloat[*df.to_matrix]
result = analyzer.fit_predict(samples)

df[:cluster_id] = result
pp df.head(5)
pp df.cluster_id.value_counts
(0..3).each { |i| pp df.filter(:row) { |r| r[:cluster_id] == i }.mean.round }