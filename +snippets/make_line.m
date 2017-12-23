function line_message = make_line(varargin)
  % make a seperate line
  %
  % Parameters:
  %   description(Optional): string or char array, ''(default)
  %   placeholder(Optional): char array of length 1,  '-'(default)
  %   line_length(Optional): positive integer, 80(default)
  %
  % Example:
  % --------------------------------------BEGIN-------------------------------------
  %
  % Copyright (C) 2017 Johnny Chen  
  % Email: johnnychen94@hotmail.com
  
  p = inputParser();
  p.addOptional('description','',@(x) ischar(x)||isstring(x));
  p.addParameter('placeholder','-',@(x) length(x)==1&&ischar(x));
  p.addParameter('line_length',80,@(x) validateattributes(x,{'numeric'},...
                                                            {'integer','positive'},'draw_lines','line_length'));
                                                          
  p.parse(varargin{:});
  placeholder = char(p.Results.placeholder);
  description = char(p.Results.description);
  line_length = p.Results.line_length;
  
  
  placeholder_length = line_length - length(description);
  lf_length = ceil(placeholder_length/2);
  rt_length = placeholder_length - lf_length;

  lf_placeholders = char(ones(1,lf_length)*placeholder);
  rt_placeholders = char(ones(1,rt_length)*placeholder);
  line_message = [lf_placeholders, description, rt_placeholders,'\n'];
end
