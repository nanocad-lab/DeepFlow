function Yq = arespredict(model, Xq)
% arespredict
% Predicts output values for the given query points Xq using an ARES model.
%
% Call:
%   Yq = arespredict(model, Xq)
%
% Input:
%   model         : ARES model
%   Xq            : Inputs of query data points (Xq(i,:)), i = 1,...,nq
%
% Output:
%   Yq            : Predicted outputs of the query data points (Yq(i)),
%                   i = 1,...,nq

% =========================================================================
% ARESLab: Adaptive Regression Splines toolbox for Matlab/Octave
% Author: Gints Jekabsons (gints.jekabsons@rtu.lv)
% URL: http://www.cs.rtu.lv/jekabsons/
%
% Copyright (C) 2009-2011  Gints Jekabsons
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% =========================================================================

% Last update: November 9, 2009

if nargin < 2
    error('Too few input arguments.');
end
X = ones(size(Xq,1),length(model.knotdims)+1);
if model.trainParams.cubic
    for i = 1 : length(model.knotdims)
       X(:,i+1) = createbasisfunction(Xq, X, model.knotdims{i}, model.knotsites{i}, ...
                  model.knotdirs{i}, model.parents(i), model.minX, model.maxX, model.t1(i,:), model.t2(i,:));
    end
else
    for i = 1 : length(model.knotdims)
       X(:,i+1) = createbasisfunction(Xq, X, model.knotdims{i}, model.knotsites{i}, ...
                  model.knotdirs{i}, model.parents(i), model.minX, model.maxX);
    end
end
Yq = X * model.coefs;
return
