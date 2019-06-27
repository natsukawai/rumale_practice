require 'rumale'

samples, labels = Rumale::Dataset.load_libsvm_file('../input/iris.scale')

ss = Rumale::ModelSelection::ShuffleSplit.new(n_splits: 1, test_size: 0.3, random_seed: 1)
train_ids, test_ids = ss.split(samples, labels)[0]

train_samples = samples[train_ids, true]
puts "train_samples.shape: #{train_samples.shape}"
# train_samples.shape: [105, 4]

train_labels = labels[train_ids]
puts "train_labels.shape: #{train_labels.shape}"
# train_labels.shape: [105]

test_samples = samples[test_ids, true]
puts "test_samples.shape: #{test_samples.shape}"
# test_samples.shape: [45, 4]

test_labels = labels[test_ids]
puts "test_labels.shape: #{test_labels.shape}"
# test_labels.shape: [45]

estimator = Rumale::LinearModel::LogisticRegression.new(reg_param: 1.0, max_iter: 1000, batch_size: 20, random_seed: 1)
estimator.fit(train_samples, train_labels)
train_results = estimator.predict(train_samples)
test_results = estimator.predict(test_samples)

evaluator = Rumale::EvaluationMeasure::Accuracy.new
puts "train accuracy: #{evaluator.score(train_labels, train_results).round(5)}"
# train accuracy: 0.8381

puts "test accuracy: #{evaluator.score(test_labels, test_results).round(5)}"
# test accuracy: 0.73333