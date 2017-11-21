function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

%m = size(z,1);
%n = size(z,2);
%sig = 1.0 ./ (1.0 + exp(-z));

%for i=1:m,
%	for j=1:n,
%		g(i,j) = 1.0 ./ (1.0 + exp(-z(i,j))) * (1 -  1.0 ./ (1.0 + exp(-z(i,j))));
%	end
%end


g = sigmoid(z) .* (1 - sigmoid(z));








% =============================================================




end
