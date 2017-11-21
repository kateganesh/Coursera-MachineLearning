function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X];

% aa = size(Theta1)

% bb = size(X) 

%cc = size(Theta2)

% a_super_1 = X;

% z_super_2 = Theta1 * a_super_1;

% a_super_2 = sigmoid(z_super_2);

% a_super_2 = a_super_2'; 

% sz = size(a_super_2,1);

% a_super_2 = [ones(sz,1) a_super_2];

% z_super_3 = Theta2 * a_super_2';

% a_super_3 = sigmoid(z_super_3);

% [pre index] = max(a_super_3, [], 2);

% p=index;

a_super_1 = X;
z_super_2 = a_super_1 * Theta1';
a_super_2 = sigmoid(z_super_2);
sz = size(a_super_2, 1);
a_super_2 = [ones(sz, 1) a_super_2];
%size(a_super_2)
z_super_3 = a_super_2 * Theta2';
a_super_3 = sigmoid(z_super_3);
%size(a_super_3)
[pred index] = max(a_super_3, [],2);
p = index;



% =========================================================================


end
