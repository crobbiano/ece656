function [ next_weights, inv_corr ] = wrls_weight_update( datamat, curr_weights, gamma, prev_inv_corr, desired, a)
    %wrls_weight_update Calculate the next weight vector for the WRLS algorithm
      
    gain = prev_inv_corr*datamat/(gamma + datamat'*prev_inv_corr*datamat);
    inv_corr = (1/gamma)*(eye(size(prev_inv_corr)) - gain*datamat')*prev_inv_corr;
    
    next_weights = curr_weights + gain*(desired - (1/a)*datamat'*curr_weights);
end

