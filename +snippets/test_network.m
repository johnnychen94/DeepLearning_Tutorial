function [counts,corrects,test_duration, wrong_imgs] = test_network(trained_net,test_imgs,test_labels)
  % test network performance
  %
  % Parameters:
  %   trained_net(Required), SeriesNetwork :  network to test
  %   test_imgs(Required), 4-D double :   test images
  %   test_labels(Required), [Nx1] categorical vector :  ground truth of test images
  %   
  % Return:
  %   counts,  [Nx1]double  :  count for image for each category, including the total count
  %   corrects,  [Nx1]double  :  count for correct prediction for each category, including the
  %       total corrects
  %   test_duration,   duration :  duration of test time
  %   wrong_imgs,  [Mx1]cell :   wrong predicted images
  %
  % Copyright (C) 2017 Johnny Chen  
  % Email: johnnychen94@hotmail.com
  
  labels = unique(test_labels);
  
  % initialize results
  test_start_time = datetime;
  counts = zeros(length(labels)+1,1);
  corrects = zeros(length(labels)+1,1);
  wrong_imgs = cell(length(test_imgs),1);
  wrong_count = 1;
  
  % testing
  p = randperm(length(test_imgs));
  for i = 1:length(test_imgs)
    cur_img = test_imgs(:,:,1,p(i));
    [~, cur_predict_index] = max(predict(trained_net,cur_img));
    cur_truth_index = find(labels==test_labels(p(i)));

    counts(cur_truth_index) = counts(cur_truth_index)+1;
    if cur_truth_index == cur_predict_index
      corrects(cur_truth_index) = corrects(cur_truth_index) +1;
    else
      wrong_imgs{wrong_count} = cur_img;
      wrong_count = wrong_count+1;
    end
  end
  
  % post process results
  counts(length(labels)+1) = sum(counts);
  corrects(length(labels)+1) = sum(corrects);
  test_duration = datetime - test_start_time;
  wrong_imgs = wrong_imgs(1:wrong_count);
end
