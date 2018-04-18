function [ z ] = calcZis( eta, x, kernel_mats, samples, ls, A, scratch_num, kappa, curr_kernel)
    %calcZis Calculates the zi's
    %   Find the current kernel which is a mixture of all ranked kernels
    %   then classifies all samples using the current kernel and
    %   zi == 1 if classification is correct
      
    full_kernel = zeros(size(A,2), size(A,2));
    single_kappa = 0;
    partial_kappa = zeros(1, size(A,2));
    samples_norm = samples/norm(samples, 2);
    for m=1:length(eta)
        K = zeros(size(A,2), size(A,2));
        for i=1:size(A,2)
            for j=1:size(A,2)
                K(i,j) = kappa{m}(A(:,i), A(:,j));
            end
            partial_kappa(i) = partial_kappa(i) +  eta(m)*kappa{m}(samples_norm, A(:,i));
        end
        full_kernel = full_kernel + eta(m)*K;
        single_kappa = single_kappa + eta(m)*kappa{m}(samples_norm,samples_norm);
    end
%     
%     full_kernel(scratch_num, :) = 0;
%     full_kernel(:, scratch_num) = 0;
%     partial_kappa(:, scratch_num) = 0;
    
    % get current kernel function
    if iscell(kernel_mats)
        curr_kernel = zeros(size(kernel_mats{1},1));
        for i=1:length(eta)
            curr_kernel = curr_kernel + eta(i)*kernel_mats{i};
        end
    else
        curr_kernel = zeros(size(kernel_mats,1));
        for i=1:length(eta)
            curr_kernel = curr_kernel + eta(i)*kernel_mats;
        end
    end
    
    % Find the number of samples in each class
    classes = unique(ls);
    num_classes = numel(classes);
    for i=1:num_classes
        num_samples_per_class(i) = sum(ls == classes(i));
    end
    
    % Find class - FIXME - need to calculate the curr_kernel(i,i) for the
    % new samples as well as curr_kernel(i, Y)
    z=zeros(1, size(samples,2));
    for i=1:size(samples,2)
        err = [];
        for class=1:num_classes
            err(class) = curr_kernel(i,i) + x(class:class + num_samples_per_class - 1,i)'*...
                curr_kernel(class:class + num_samples_per_class - 1,class:class + num_samples_per_class - 1)*x(class:class + num_samples_per_class - 1,i)...
                - 2*curr_kernel(i,class:class + num_samples_per_class - 1)*x(class:class + num_samples_per_class - 1,i);
        end
        [~, z(i)] = min(err);
    end
end

