function ret = isstring(obj)
%
% Returns true if the argument is a row vector of characters.
%
ret = ischar(obj) & ndims(obj) == 2 & size(obj,1) == 1;
