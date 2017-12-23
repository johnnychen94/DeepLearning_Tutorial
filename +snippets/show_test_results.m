function show_test_results(counts,corrects,labels,test_duration)
  % Show test results
  %
  % Parameters:
  %   counts(Required): [(N+1)x1]   numerical
  %   corrects(Required): [(N+1)x1]   numerical
  %   labels(Required): [Nx1] categorical vector
  %   test_duration(Required): duration
  %
  % Example:
  %   ----------------------------------Test Results----------------------------------
  % Category:	0	Counts:	980	Corrects:	974	Accuracy:	 99.3878 %
  % Category:	1	Counts:	1135	Corrects:	1128	Accuracy:	 99.3833 %
  % Category:	2	Counts:	1032	Corrects:	1020	Accuracy:	 98.8372 %
  % Category:	3	Counts:	1010	Corrects:	999	Accuracy:	 98.9109 %
  % Category:	4	Counts:	982	Corrects:	972	Accuracy:	 98.9817 %
  % Category:	5	Counts:	892	Corrects:	878	Accuracy:	 98.4305 %
  % Category:	6	Counts:	958	Corrects:	939	Accuracy:	 98.0167 %
  % Category:	7	Counts:	1028	Corrects:	1009	Accuracy:	 98.1518 %
  % Category:	8	Counts:	974	Corrects:	959	Accuracy:	 98.4600 %
  % Category:	9	Counts:	1009	Corrects:	986	Accuracy:	 97.7205 %
  % Total:			Counts:	10000	Corrects:	9864	Accuracy:	 98.6400 %
  % Duration:	00:00:19	FPS:	523.8 image/s
  % --------------------------------------------------------------------------------
  %
  % Copyright (C) 2017 Johnny Chen  
  % Email: johnnychen94@hotmail.com
  
  make_line = @snippets.make_line; % NOTE: change this if you want to copy this function
  accuracies = 100*corrects./counts;
  
  % make header
  fprintf(make_line('Test Results'))
  
  % result for each category
  for i = 1:length(labels)
    fprintf("Category:\t"+string(labels(i))+...
      "\tCounts:\t"+string(counts(i))+...
      "\tCorrects:\t"+string(corrects(i))+...
      "\tAccuracy:\t %.4f %%"+"\n",accuracies(i));
  end
  
  % show total mean result 
  fprintf("Total:\t\t"+...
      "\tCounts:\t"+string(counts(length(labels)+1))+...
      "\tCorrects:\t"+string(corrects(length(labels)+1))+...
      "\tAccuracy:\t %.4f %%"+"\n",accuracies(length(labels)+1));
    
  % show speed: duration, fps
  fprintf("Duration:\t"+string(test_duration)+...
      "\tFPS:\t%.1f image/s"+"\n",counts(length(labels)+1)/seconds(test_duration));
    
  % make footer
  fprintf(make_line());
end
