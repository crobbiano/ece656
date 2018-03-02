function [ next_weights, inv_corr ] = wrls_weight_update( datamat, curr_weights, gamma, prev_inv_corr, desired, a)
    %wrls_weight_update Calculate the next weight vector for the WRLS algorithm
      
    mm = size(curr_weights,1);
%     P = .5*eye(mm);
    inv_corr = prev_inv_corr;
    next_weights = curr_weights;
    for n=1:size(datamat,2)
        gain = inv_corr*datamat(:,n)/(gamma + datamat(:,n)'*inv_corr*datamat(:,n));
        inv_corr = (1/gamma)*(eye(mm) - gain*datamat(:,n)')*inv_corr;
        next_weights = next_weights + gain*(desired(n) - (1/a)*datamat(:,n)'*next_weights);
    end
end

