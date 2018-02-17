function [ next_weights ] = lms_weight_update( datamat, curr_weights, desired, learning_rate)
    %lms_weight_update Calculate the next weight vector for the LMS/SGD algorithm
    %   Using the cost function J(w(k))=e_i(k)^2=(d_i(k) -w_i(k)'x(k))^2 
    %   and the update rule w_i(k+1) = w_i(k) + mu*x(k)*e_i(k), we calculate
    %   w(k+1) using the sample correlation and cross correlations.  This
    %   makes w_i(k+1)= w_i(k) + mu*sum_l(x(l)*e_i(l))
    
    thing = curr_weights'*datamat;
    e = desired - curr_weights'*datamat; 
    thing2 = learning_rate*datamat*e';
    next_weights = curr_weights + learning_rate*datamat*e';
end

